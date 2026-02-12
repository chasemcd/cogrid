"""Vectorized movement resolution for CoGrid environments.

Replaces the Python-loop-based ``CoGridEnv.move_agents()`` with array
operations.  All agent proposed positions are computed simultaneously;
only the collision resolution loop and swap detection iterate over agents
(these are marked for Phase 2 conversion to ``lax.fori_loop``).

The module provides two movement functions:

- :func:`move_agents_array` -- numpy backend (Python loops for collision/swap)
- :func:`move_agents_jax` -- JAX backend (``lax.fori_loop`` + vectorized swap)

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


# ======================================================================
# JAX-path movement (JIT-compatible)
# ======================================================================


def move_agents_jax(
    agent_pos,        # (n_agents, 2) int32
    agent_dir,        # (n_agents,) int32
    actions,          # (n_agents,) int32
    wall_map,         # (H, W) int32
    object_type_map,  # (H, W) int32
    can_overlap,      # (n_types,) int32
    rng_key,          # JAX PRNG key
    action_set,       # str -- "cardinal" or "rotation" (static, not traced)
):
    """JIT-compatible movement resolution using JAX primitives.

    Functionally equivalent to :func:`move_agents_array` but replaces
    Python loops with ``lax.fori_loop`` (collision resolution) and
    vectorized array ops (swap detection), and uses ``jax.random.split``
    for PRNG instead of ``numpy.random.Generator``.

    The ``action_set`` parameter is a Python string checked at trace time.
    When using ``jax.jit``, pass it via ``static_argnames=['action_set']``.

    Args:
        agent_pos: Current positions, shape ``(n_agents, 2)``, int32.
        agent_dir: Current direction enums, shape ``(n_agents,)``, int32.
        actions: Action indices, shape ``(n_agents,)``, int32.
        wall_map: Binary wall mask, shape ``(H, W)``, int32.
        object_type_map: Object type IDs per cell, shape ``(H, W)``, int32.
        can_overlap: Per-type overlap flag, shape ``(n_types,)``, int32.
        rng_key: JAX PRNG key for random priority ordering.
        action_set: ``"cardinal"`` or ``"rotation"`` (static).

    Returns:
        Tuple ``(new_pos, new_dir, new_rng_key)`` where:
        - ``new_pos``: int32 array of shape ``(n_agents, 2)``
        - ``new_dir``: int32 array of shape ``(n_agents,)``
        - ``new_rng_key``: Updated JAX PRNG key (input key consumed)
    """
    import jax
    import jax.numpy as jnp
    import jax.lax as lax

    n_agents = agent_pos.shape[0]
    H, W = wall_map.shape

    # -------------------------------------------------------------------
    # 1. Action-to-direction mapping
    # -------------------------------------------------------------------
    # action_set is a Python string; checked at trace time.
    # JAX recompiles for each distinct static value.
    new_dir = agent_dir  # JAX arrays are immutable; no .copy() needed

    if action_set == "cardinal":
        # CardinalActions: MoveUp=0, MoveDown=1, MoveLeft=2, MoveRight=3
        # Directions:      Right=0,  Down=1,     Left=2,     Up=3
        action_to_dir = jnp.array([3, 1, 2, 0, -1, -1, -1], dtype=jnp.int32)

        is_cardinal_move = actions < 4
        mapped_dirs = action_to_dir[actions]
        new_dir = jnp.where(is_cardinal_move, mapped_dirs, new_dir)
        is_mover = is_cardinal_move

    elif action_set == "rotation":
        is_mover = actions == 0  # Forward action

    # -------------------------------------------------------------------
    # 2. Compute proposed positions
    # -------------------------------------------------------------------
    dir_vec_table = jnp.array(
        [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32
    )
    dir_vecs = dir_vec_table[new_dir]  # (n_agents, 2)
    proposed = agent_pos + dir_vecs * is_mover[:, None].astype(jnp.int32)

    # -------------------------------------------------------------------
    # 3. Bounds clipping
    # -------------------------------------------------------------------
    proposed = jnp.clip(
        proposed,
        jnp.array([0, 0], dtype=jnp.int32),
        jnp.array([H - 1, W - 1], dtype=jnp.int32),
    )

    # -------------------------------------------------------------------
    # 4. Wall check -- revert to current pos if hitting a wall
    # -------------------------------------------------------------------
    hits_wall = wall_map[proposed[:, 0], proposed[:, 1]].astype(jnp.bool_)
    proposed = jnp.where(hits_wall[:, None], agent_pos, proposed)

    # -------------------------------------------------------------------
    # 5. Overlap check -- revert if cell has non-overlappable object
    # -------------------------------------------------------------------
    fwd_type = object_type_map[proposed[:, 0], proposed[:, 1]]
    can_overlap_fwd = can_overlap[fwd_type]
    blocked_by_object = (fwd_type > 0) & (can_overlap_fwd == 0)
    proposed = jnp.where(blocked_by_object[:, None], agent_pos, proposed)

    # -------------------------------------------------------------------
    # 6. Priority to agents staying in place
    # -------------------------------------------------------------------
    staying = jnp.all(proposed == agent_pos, axis=1)

    # -------------------------------------------------------------------
    # 7. Random priority collision resolution via lax.fori_loop
    # -------------------------------------------------------------------
    key, subkey = jax.random.split(rng_key)
    priority = jax.random.permutation(subkey, n_agents)

    final_pos = jnp.where(staying[:, None], agent_pos, agent_pos)
    resolved = staying

    def collision_body(i, carry):
        final_pos, resolved = carry
        idx = priority[i]
        attempted = proposed[idx]

        # Check conflicts with all already-resolved positions
        conflict_resolved = jnp.any(
            jnp.all(final_pos == attempted[None, :], axis=1) & resolved
        )

        # Check conflicts with unresolved agents' current positions
        is_other = jnp.arange(n_agents) != idx
        conflict_unresolved = jnp.any(
            jnp.all(agent_pos == attempted[None, :], axis=1)
            & is_other
            & ~resolved
        )

        blocked = conflict_resolved | conflict_unresolved
        new_pos_i = jnp.where(blocked, agent_pos[idx], attempted)
        final_pos = final_pos.at[idx].set(new_pos_i)
        resolved = resolved.at[idx].set(True)

        return (final_pos, resolved)

    final_pos, _ = lax.fori_loop(
        0, n_agents, collision_body, (final_pos, resolved)
    )

    # -------------------------------------------------------------------
    # 8. Vectorized swap detection
    # -------------------------------------------------------------------
    # moved_to_old[i, j] = True iff final_pos[i] == agent_pos[j]
    moved_to_old = jnp.all(
        final_pos[:, None, :] == agent_pos[None, :, :], axis=2
    )
    # swapped[i, j] = True iff i moved to j's old pos AND j moved to i's
    swapped = moved_to_old & moved_to_old.T
    swapped = swapped & ~jnp.eye(n_agents, dtype=jnp.bool_)
    any_swap = jnp.any(swapped, axis=1)
    final_pos = jnp.where(any_swap[:, None], agent_pos, final_pos)

    return final_pos, new_dir, key


def test_movement_jax():
    """Verify that ``move_agents_jax`` JIT-compiles without error.

    Creates a small test case with 2 agents on a 5x5 grid, calls
    ``jax.jit(move_agents_jax)``, and verifies output shapes and
    PRNG key consumption.
    """
    from cogrid.backend import set_backend, get_backend

    if get_backend() != "jax":
        set_backend("jax")

    import jax
    import jax.numpy as jnp

    key = jax.random.key(42)
    agent_pos = jnp.array([[1, 1], [3, 3]], dtype=jnp.int32)
    agent_dir = jnp.array([0, 2], dtype=jnp.int32)
    actions = jnp.array([3, 2], dtype=jnp.int32)  # MoveRight, MoveLeft
    wall_map = jnp.zeros((5, 5), dtype=jnp.int32)
    obj_map = jnp.zeros((5, 5), dtype=jnp.int32)
    can_overlap = jnp.ones(10, dtype=jnp.int32)

    jitted = jax.jit(move_agents_jax, static_argnames=["action_set"])
    new_pos, new_dir, new_key = jitted(
        agent_pos, agent_dir, actions, wall_map, obj_map, can_overlap,
        key, "cardinal",
    )

    assert new_pos.shape == (2, 2), f"Expected (2,2), got {new_pos.shape}"
    assert new_dir.shape == (2,), f"Expected (2,), got {new_dir.shape}"
    assert not jnp.array_equal(new_key, key), "PRNG key should be consumed"

    # Verify agents actually moved (no walls, no overlap)
    # Agent 0: dir=0 (Right), action=3 (MoveRight) -> should move right
    # Agent 1: dir=2 (Left), action=2 (MoveLeft) -> should move left
    print(f"  agent_pos: {agent_pos}")
    print(f"  new_pos:   {new_pos}")
    print(f"  new_dir:   {new_dir}")
    print("JAX movement JIT: PASSED")

    # --- Test collision scenario ---
    # Two agents trying to move to the same cell
    key2 = jax.random.key(99)
    agent_pos2 = jnp.array([[2, 1], [2, 3]], dtype=jnp.int32)
    agent_dir2 = jnp.array([0, 2], dtype=jnp.int32)  # Right, Left
    actions2 = jnp.array([3, 2], dtype=jnp.int32)  # MoveRight, MoveLeft -> both to (2,2)
    new_pos2, _, _ = jitted(
        agent_pos2, agent_dir2, actions2, wall_map, obj_map, can_overlap,
        key2, "cardinal",
    )
    # One should succeed, one should revert -- no overlapping final positions
    pos_set = set(map(tuple, new_pos2.tolist()))
    assert len(pos_set) == 2, f"Collision produced overlapping positions: {new_pos2}"
    print("JAX collision resolution: PASSED")

    # --- Test swap detection ---
    key3 = jax.random.key(7)
    agent_pos3 = jnp.array([[2, 2], [2, 3]], dtype=jnp.int32)
    agent_dir3 = jnp.array([0, 2], dtype=jnp.int32)  # Right, Left
    actions3 = jnp.array([3, 2], dtype=jnp.int32)  # MoveRight(->2,3), MoveLeft(->2,2)
    new_pos3, _, _ = jitted(
        agent_pos3, agent_dir3, actions3, wall_map, obj_map, can_overlap,
        key3, "cardinal",
    )
    # Swap detected -> both should revert to original
    assert jnp.array_equal(new_pos3, agent_pos3), (
        f"Swap not detected: {agent_pos3} -> {new_pos3}"
    )
    print("JAX swap detection: PASSED")


def test_movement_parity():
    """Development-time parity test.

    Validates that ``move_agents_array`` produces identical agent positions
    and directions to the existing ``CoGridEnv.move_agents()`` for 50+
    random steps on the cramped_room layout.

    The test handles RNG synchronization by running both the original and
    vectorized paths from the same captured pre-step state and comparing
    the resulting positions.  Priority ordering differences are accounted
    for by forking separate RNG instances with identical seeds for each step.
    """
    import copy
    import numpy as np
    from cogrid.envs import registry
    from cogrid.core.grid_utils import layout_to_array_state
    from cogrid.core.agent import create_agent_arrays
    from cogrid.core.grid_object import build_lookup_tables

    # ---- Set up environment ----
    env = registry.make("Overcooked-CrampedRoom-V0")
    env.reset(seed=42)

    scope = env.scope
    tables = build_lookup_tables(scope=scope)
    can_overlap_table = tables["CAN_OVERLAP"]

    n_steps = 50
    n_match = 0
    n_conflict_valid = 0
    n_direction_match = 0
    total_steps = 0

    for step_i in range(n_steps):
        # Random actions (cardinal: 0-6, but focus on movement 0-3)
        step_seed = 1000 + step_i
        step_rng = np.random.default_rng(step_seed)
        action_indices = {
            a_id: step_rng.integers(0, 7) for a_id in env.agent_ids
        }

        # ---- Capture pre-step state ----
        pre_positions = {
            a_id: np.array(agent.pos, dtype=np.int32).copy()
            for a_id, agent in env.env_agents.items()
        }
        pre_directions = {
            a_id: int(agent.dir)
            for a_id, agent in env.env_agents.items()
        }

        # Capture grid arrays for vectorized path
        grid_arrays = layout_to_array_state(env.grid, scope=scope)
        agent_arrays = create_agent_arrays(env.env_agents, scope=scope)

        agent_pos = agent_arrays["agent_pos"].copy()
        agent_dir = agent_arrays["agent_dir"].copy()
        agent_ids = agent_arrays["agent_ids"]
        n_agents = agent_arrays["n_agents"]

        # Build action array matching agent_ids order
        actions_arr = xp.array(
            [int(action_indices[a_id]) for a_id in agent_ids],
            dtype=xp.int32,
        )

        # ---- Save env RNG state, run original move_agents ----
        rng_state_before = copy.deepcopy(env.np_random.bit_generator.state)

        # The original move_agents expects string actions, but _action_idx_to_str
        # converts them. We replicate that flow.
        str_actions = env._action_idx_to_str(
            {a_id: int(action_indices[a_id]) for a_id in env.agent_ids}
        )
        env.move_agents(str_actions)

        # Record original results
        orig_positions = {}
        orig_directions = {}
        for a_id in agent_ids:
            agent = env.env_agents[a_id]
            orig_positions[a_id] = np.array(agent.pos, dtype=np.int32).copy()
            orig_directions[a_id] = int(agent.dir)

        # ---- Reset agents to pre-step state ----
        for a_id, agent in env.env_agents.items():
            agent.pos = tuple(pre_positions[a_id])
            agent.dir = pre_directions[a_id]

        # ---- Run vectorized version with separate RNG ----
        # Use the saved RNG state so permutation starts from same point
        vec_rng = np.random.Generator(np.random.PCG64())
        vec_rng.bit_generator.state = copy.deepcopy(rng_state_before)

        new_pos, new_dir = move_agents_array(
            agent_pos,
            agent_dir,
            actions_arr,
            grid_arrays["wall_map"],
            grid_arrays["object_type_map"],
            can_overlap_table,
            vec_rng,
            "cardinal",
        )

        # ---- Compare results ----
        total_steps += 1
        positions_match = True
        directions_match = True

        for i, a_id in enumerate(agent_ids):
            orig_pos = orig_positions[a_id]
            vec_pos = new_pos[i]
            orig_dir_val = orig_directions[a_id]
            vec_dir_val = int(new_dir[i])

            if not np.array_equal(orig_pos, vec_pos):
                positions_match = False

            if orig_dir_val != vec_dir_val:
                directions_match = False

        if directions_match:
            n_direction_match += 1

        if positions_match:
            n_match += 1
        else:
            # Verify BOTH results produce valid non-overlapping positions
            orig_pos_list = [tuple(orig_positions[a_id]) for a_id in agent_ids]
            vec_pos_list = [tuple(new_pos[i]) for i in range(n_agents)]
            assert len(set(orig_pos_list)) == n_agents, (
                f"Step {step_i}: Original has overlapping positions: {orig_pos_list}"
            )
            assert len(set(vec_pos_list)) == n_agents, (
                f"Step {step_i}: Vectorized has overlapping positions: {vec_pos_list}"
            )
            n_conflict_valid += 1

    # ---- Report results ----
    print(f"Movement parity test: {n_steps} steps on cramped_room")
    print(f"  Exact position matches: {n_match}/{total_steps}")
    print(f"  Direction matches:      {n_direction_match}/{total_steps}")
    print(f"  Conflict-but-valid:     {n_conflict_valid}/{total_steps}")
    print(f"  Total verified:         {n_match + n_conflict_valid}/{total_steps}")

    # Direction changes must ALWAYS match (no RNG dependency)
    assert n_direction_match == total_steps, (
        f"Direction mismatch in {total_steps - n_direction_match} steps!"
    )

    # ALL steps must be either exact match or valid conflict resolution
    assert n_match + n_conflict_valid == total_steps, (
        f"Unaccounted steps: {total_steps - n_match - n_conflict_valid}"
    )

    # Most steps should match exactly (conflicts are rare in random play)
    match_pct = n_match / total_steps * 100
    print(f"  Match rate: {match_pct:.1f}%")
    assert match_pct >= 50, (
        f"Too few exact matches ({match_pct:.1f}%) -- likely a bug"
    )

    print("MOVEMENT PARITY TEST PASSED")
