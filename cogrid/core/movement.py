"""Vectorized movement resolution for CoGrid environments.

Replaces the Python-loop-based ``CoGridEnv.move_agents()`` with array
operations.  All agent proposed positions are computed simultaneously;
only the collision resolution loop and swap detection iterate over agents
(these are marked for Phase 2 conversion to ``lax.fori_loop``).

Usage::

    from cogrid.core.movement import move_agents_array

    new_pos, new_dir = move_agents_array(
        agent_pos, agent_dir, actions,
        wall_map, object_type_map, can_overlap,
        rng, action_set,
    )
"""

from __future__ import annotations

from cogrid.backend import xp
from cogrid.core.agent import get_dir_vec_table


# Cardinal action indices -> direction enum mapping.
# CardinalActions order: MoveUp(0), MoveDown(1), MoveLeft(2), MoveRight(3),
#                        PickupDrop(4), Toggle(5), Noop(6)
# Directions: Right=0, Down=1, Left=2, Up=3
# Non-movement actions (4-6) mapped to -1 sentinel.
ACTION_TO_DIR = None  # Lazily initialized


def _get_action_to_dir():
    """Return the ACTION_TO_DIR mapping array, creating it lazily."""
    global ACTION_TO_DIR
    if ACTION_TO_DIR is None:
        ACTION_TO_DIR = xp.array([3, 1, 2, 0, -1, -1, -1], dtype=xp.int32)
    return ACTION_TO_DIR


def move_agents_array(
    agent_pos,        # (n_agents, 2) int32 -- current positions [row, col]
    agent_dir,        # (n_agents,) int32 -- current directions
    actions,          # (n_agents,) int32 -- action indices
    wall_map,         # (H, W) int32 -- 1 where walls exist
    object_type_map,  # (H, W) int32 -- type IDs at each cell
    can_overlap,      # (n_types,) int32 -- 1 if overlappable, 0 if not
    rng,              # numpy.random.Generator -- for priority shuffle
    action_set,       # str -- "cardinal" or "rotation"
):
    """Compute new agent positions and directions from actions.

    This is the vectorized equivalent of ``CoGridEnv.move_agents()``.
    All proposed positions are computed via array operations.  The collision
    resolution loop and swap detection iterate over agents but are marked
    for Phase 2 conversion to ``lax.fori_loop``.

    Args:
        agent_pos: Current positions, shape ``(n_agents, 2)``, int32.
        agent_dir: Current direction enums, shape ``(n_agents,)``, int32.
        actions: Action indices, shape ``(n_agents,)``, int32.
        wall_map: Binary wall mask, shape ``(H, W)``, int32.
        object_type_map: Object type IDs per cell, shape ``(H, W)``, int32.
        can_overlap: Per-type overlap flag, shape ``(n_types,)``, int32.
        rng: ``numpy.random.Generator`` for random priority ordering.
        action_set: ``"cardinal"`` or ``"rotation"``.

    Returns:
        Tuple ``(new_pos, new_dir)`` where:
        - ``new_pos``: int32 array of shape ``(n_agents, 2)``
        - ``new_dir``: int32 array of shape ``(n_agents,)``
    """
    n_agents = agent_pos.shape[0]
    H, W = wall_map.shape

    # Work on copies so we don't mutate the caller's arrays
    new_dir = agent_dir.copy()
    is_mover = xp.zeros(n_agents, dtype=xp.bool_)

    # -------------------------------------------------------------------
    # 1. Action-to-direction mapping
    # -------------------------------------------------------------------
    if action_set == "cardinal":
        action_to_dir = _get_action_to_dir()

        # Identify which agents are taking a cardinal movement action (indices 0-3)
        is_cardinal_move = actions < 4  # MoveUp=0, MoveDown=1, MoveLeft=2, MoveRight=3

        # Update direction for cardinal movers
        # For non-movers, action_to_dir[action] would be -1, but we mask with is_cardinal_move
        mapped_dirs = action_to_dir[actions]
        new_dir = xp.where(is_cardinal_move, mapped_dirs, new_dir)

        # Cardinal movers are movers
        is_mover = is_cardinal_move

    elif action_set == "rotation":
        # In rotation mode, only Forward (action index 0) is a movement action.
        # RotateLeft/RotateRight are handled in interact(), not here.
        is_mover = actions == 0  # Forward action

    # -------------------------------------------------------------------
    # 2. Compute proposed positions
    # -------------------------------------------------------------------
    DIR_VEC_TABLE = get_dir_vec_table()
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
    # 6. Priority to agents staying in place
    # -------------------------------------------------------------------
    staying = xp.all(proposed == agent_pos, axis=1)

    # -------------------------------------------------------------------
    # 7. Random priority collision resolution
    # PHASE2: convert to lax.fori_loop or lax.scan
    # -------------------------------------------------------------------
    final_pos = agent_pos.copy()
    resolved = staying.copy()

    # Staying agents have their position finalized already
    final_pos[staying] = agent_pos[staying]

    # Random priority order
    priority = rng.permutation(n_agents)

    for idx in priority:
        if resolved[idx]:
            continue

        attempted = proposed[idx]

        # Check against all already-resolved positions
        conflict_with_resolved = False
        for j in range(n_agents):
            if resolved[j] and final_pos[j, 0] == attempted[0] and final_pos[j, 1] == attempted[1]:
                conflict_with_resolved = True
                break

        # Check against current positions of still-unresolved agents
        # (matching existing behavior: lines 499-505 in cogrid_env.py)
        conflict_with_unresolved = False
        if not conflict_with_resolved:
            for j in range(n_agents):
                if j != idx and not resolved[j]:
                    if agent_pos[j, 0] == attempted[0] and agent_pos[j, 1] == attempted[1]:
                        conflict_with_unresolved = True
                        break

        if conflict_with_resolved or conflict_with_unresolved:
            final_pos[idx] = agent_pos[idx]
        else:
            final_pos[idx] = attempted

        resolved[idx] = True

    # -------------------------------------------------------------------
    # 8. Swap detection -- revert agents that passed through each other
    # PHASE2: convert to vectorized swap detection
    # -------------------------------------------------------------------
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if (final_pos[i, 0] == agent_pos[j, 0] and
                    final_pos[i, 1] == agent_pos[j, 1] and
                    final_pos[j, 0] == agent_pos[i, 0] and
                    final_pos[j, 1] == agent_pos[i, 1]):
                final_pos[i] = agent_pos[i]
                final_pos[j] = agent_pos[j]

    return final_pos, new_dir
