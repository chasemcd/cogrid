"""Vectorized movement resolution for CoGrid environments.

Provides a single :func:`move_agents` function that computes new agent
positions and directions using pure ``xp`` array operations. Collision
resolution and swap detection are fully vectorized via pairwise conflict
matrices and priority masking -- no Python loops, no ``lax.fori_loop``.

The caller supplies a pre-computed ``priority`` array (agent indices in
resolution order), keeping this module free of backend-specific RNG logic.

Usage::

    from cogrid.core.movement import move_agents

    new_pos, new_dir = move_agents(
        agent_pos, agent_dir, actions,
        wall_map, object_type_map, can_overlap,
        priority, action_set,
    )
"""

from __future__ import annotations

from cogrid.backend import xp


def move_agents(
    agent_pos,        # (n_agents, 2) int32 -- current positions [row, col]
    agent_dir,        # (n_agents,) int32 -- current directions
    actions,          # (n_agents,) int32 -- action indices
    wall_map,         # (H, W) int32 -- 1 where walls exist
    object_type_map,  # (H, W) int32 -- type IDs at each cell
    can_overlap,      # (n_types,) int32 -- 1 if overlappable, 0 if not
    priority,         # (n_agents,) int32 -- pre-computed priority ordering
    action_set,       # str -- "cardinal" or "rotation"
):
    """Compute new agent positions and directions from actions.

    All proposed positions, collision resolution, and swap detection are
    computed via ``xp`` array operations. The function works identically
    on numpy and JAX backends.

    Args:
        agent_pos: Current positions, shape ``(n_agents, 2)``, int32.
        agent_dir: Current direction enums, shape ``(n_agents,)``, int32.
        actions: Action indices, shape ``(n_agents,)``, int32.
        wall_map: Binary wall mask, shape ``(H, W)``, int32.
        object_type_map: Object type IDs per cell, shape ``(H, W)``, int32.
        can_overlap: Per-type overlap flag, shape ``(n_types,)``, int32.
        priority: Agent indices in resolution order, shape ``(n_agents,)``,
            int32. ``priority[0]`` is the highest-priority agent. The caller
            generates this via ``rng.permutation(n_agents)`` (numpy) or
            ``jax.random.permutation(key, n_agents)`` (JAX).
        action_set: ``"cardinal"`` or ``"rotation"``.

    Returns:
        Tuple ``(new_pos, new_dir)`` where:
        - ``new_pos``: int32 array of shape ``(n_agents, 2)``
        - ``new_dir``: int32 array of shape ``(n_agents,)``
    """
    n_agents = agent_pos.shape[0]
    H, W = wall_map.shape

    # -------------------------------------------------------------------
    # 1. Action-to-direction mapping
    # -------------------------------------------------------------------
    new_dir = agent_dir  # no .copy() -- use xp.where to build result
    is_mover = xp.zeros(n_agents, dtype=xp.bool_)

    if action_set == "cardinal":
        # CardinalActions: MoveUp=0, MoveDown=1, MoveLeft=2, MoveRight=3
        # Directions:      Right=0,  Down=1,     Left=2,     Up=3
        action_to_dir = xp.array([3, 1, 2, 0, -1, -1, -1], dtype=xp.int32)

        is_cardinal_move = actions < 4
        mapped_dirs = action_to_dir[actions]
        new_dir = xp.where(is_cardinal_move, mapped_dirs, new_dir)
        is_mover = is_cardinal_move

    elif action_set == "rotation":
        is_mover = actions == 0  # Forward action

    # -------------------------------------------------------------------
    # 2. Compute proposed positions
    # -------------------------------------------------------------------
    DIR_VEC_TABLE = xp.array(
        [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=xp.int32
    )
    dir_vecs = DIR_VEC_TABLE[new_dir]  # (n_agents, 2)
    proposed = agent_pos + dir_vecs * is_mover[:, None].astype(xp.int32)

    # -------------------------------------------------------------------
    # 3. Bounds clipping
    # -------------------------------------------------------------------
    proposed = xp.clip(proposed, xp.array([0, 0], dtype=xp.int32),
                       xp.array([H - 1, W - 1], dtype=xp.int32))

    # -------------------------------------------------------------------
    # 4. Wall check -- revert to current pos if hitting a wall
    # -------------------------------------------------------------------
    hits_wall = wall_map[proposed[:, 0], proposed[:, 1]].astype(xp.bool_)
    proposed = xp.where(hits_wall[:, None], agent_pos, proposed)

    # -------------------------------------------------------------------
    # 5. Overlap check -- revert if cell has non-overlappable object
    # -------------------------------------------------------------------
    fwd_type = object_type_map[proposed[:, 0], proposed[:, 1]]
    can_overlap_fwd = can_overlap[fwd_type]
    blocked_by_object = (fwd_type > 0) & (can_overlap_fwd == 0)
    proposed = xp.where(blocked_by_object[:, None], agent_pos, proposed)

    # -------------------------------------------------------------------
    # 6. Identify agents staying in place
    # -------------------------------------------------------------------
    staying = xp.all(proposed == agent_pos, axis=1)

    # -------------------------------------------------------------------
    # 7. Vectorized collision resolution
    #
    # Replaces both the Python for-loop (numpy path) and lax.fori_loop
    # (JAX path) with pairwise conflict detection and priority masking.
    #
    # For each agent, determine if it is "blocked" from reaching its
    # proposed position. An agent is blocked if:
    #   (a) Another agent proposes the same cell and has higher priority
    #   (b) It proposes a cell occupied by a staying agent
    #   (c) It proposes a cell occupied by an unresolved (lower-priority)
    #       agent whose current position is still "claimed"
    #
    # After the initial pass, blocked agents stay at their current
    # positions. A cascade pass then blocks any agent whose proposed
    # position matches a newly-blocked agent's current position. For
    # n_agents <= 4 (typical grid-world), one cascade pass suffices.
    # -------------------------------------------------------------------

    # Priority rank: rank[i] = position of agent i in priority order
    # (0 = highest priority, resolves first). Double argsort gives rank.
    rank = xp.argsort(xp.argsort(priority))

    # Pairwise same-proposed check: (n_agents, n_agents) bool
    # same_proposed[i, j] = True iff proposed[i] == proposed[j] and i != j
    same_proposed = xp.all(
        proposed[:, None, :] == proposed[None, :, :], axis=2
    )
    same_proposed = same_proposed & ~xp.eye(n_agents, dtype=xp.bool_)

    # Pairwise proposed-into-current check: (n_agents, n_agents) bool
    # into_current[i, j] = True iff proposed[i] == agent_pos[j] and i != j
    into_current = xp.all(
        proposed[:, None, :] == agent_pos[None, :, :], axis=2
    )
    into_current = into_current & ~xp.eye(n_agents, dtype=xp.bool_)

    # (a) Blocked by same target: agent j has higher priority (lower rank)
    higher_priority = rank[None, :] < rank[:, None]  # [i, j]: j has higher priority than i
    blocked_by_higher = xp.any(same_proposed & higher_priority, axis=1)

    # (b) Blocked by stationary agent: proposed[i] == agent_pos[j] where j is staying
    blocked_by_staying = xp.any(into_current & staying[None, :], axis=1)

    # (c) Blocked by unresolved agent: proposed[i] == agent_pos[j] where
    # j is NOT staying AND j has lower priority (higher rank) than i.
    # When agent i resolves, agents with higher rank haven't resolved yet,
    # so their current positions are still "claimed".
    lower_priority = rank[None, :] > rank[:, None]  # [i, j]: j has lower priority than i
    blocked_by_unresolved = xp.any(
        into_current & ~staying[None, :] & lower_priority, axis=1
    )

    # Initial block mask (staying agents are never blocked -- they don't move)
    blocked = (blocked_by_higher | blocked_by_staying | blocked_by_unresolved) & ~staying

    # Cascade pass: agents blocked in the initial pass stay at agent_pos.
    # Any unblocked agent whose proposed position matches a blocked
    # agent's current position is also blocked. One pass suffices for
    # n_agents <= 4 because each cascade can block at most n_agents-1
    # additional agents.
    blocked_by_cascade = xp.any(into_current & blocked[None, :], axis=1) & ~staying
    blocked = blocked | blocked_by_cascade

    final_pos = xp.where(blocked[:, None], agent_pos, proposed)

    # -------------------------------------------------------------------
    # 8. Vectorized swap detection
    #
    # Two agents "swap" if each moved to the other's original position.
    # Revert both to their original positions.
    # -------------------------------------------------------------------
    moved_to_old = xp.all(
        final_pos[:, None, :] == agent_pos[None, :, :], axis=2
    )
    swapped = moved_to_old & moved_to_old.T
    swapped = swapped & ~xp.eye(n_agents, dtype=xp.bool_)
    any_swap = xp.any(swapped, axis=1)
    final_pos = xp.where(any_swap[:, None], agent_pos, final_pos)

    return final_pos, new_dir
