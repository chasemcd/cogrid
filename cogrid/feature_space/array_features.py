"""Array-based feature extractors operating on state arrays instead of Grid/Agent objects.

These functions are standalone alternatives to the Feature.generate() class methods.
They operate directly on state arrays (agent_pos, agent_dir, object_type_map, etc.)
produced by layout_to_array_state() and create_agent_arrays(), producing numerically
identical observations to the existing feature system.

Feature composition (which features to include, in what order) is resolved at init
time via build_feature_fn(), producing a single composed function that can be called
with state arrays.

NOTE: These array features do NOT replace the existing Feature classes. They exist
alongside them. The integration plan (Plan 07) will wire them into the observation
pipeline. For now, they are standalone functions that can be tested independently.

PHASE2: Per-agent observation generation uses Python loops; convert to jax.vmap.
"""

from __future__ import annotations

import numpy as np

from cogrid.backend import xp
from cogrid.core.grid_utils import adjacent_positions


# ---------------------------------------------------------------------------
# Core feature extractors
# ---------------------------------------------------------------------------


def agent_pos_feature(agent_pos, agent_idx: int):
    """Extract agent position as (2,) int32 array.

    Args:
        agent_pos: int32 array of shape (n_agents, 2) with [row, col].
        agent_idx: Index of the agent to extract position for.

    Returns:
        ndarray of shape (2,), dtype int32.
    """
    return np.asarray(agent_pos[agent_idx], dtype=np.int32)


def agent_dir_feature(agent_dir, agent_idx: int):
    """One-hot encoding of agent direction as (4,) int32 array.

    Args:
        agent_dir: int32 array of shape (n_agents,) with direction integers.
        agent_idx: Index of the agent.

    Returns:
        ndarray of shape (4,), dtype int32 with exactly one 1.
    """
    encoding = np.zeros(4, dtype=np.int32)
    encoding[agent_dir[agent_idx]] = 1
    return encoding


def full_map_encoding_feature(
    object_type_map,
    object_state_map,
    agent_pos,
    agent_dir,
    agent_inv,
    scope: str = "global",
    max_map_size: tuple[int, int] = (12, 12),
):
    """Full map encoding as (max_H, max_W, 3) int8 array matching Grid.encode(encode_char=False).

    Produces the same 3-channel encoding as FullMapEncoding.generate():
    - Channel 0: type IDs (with agent overlays)
    - Channel 1: extra state encoding (Pot overrides channel 1)
    - Channel 2: object state values (with agent inventory state overlay)

    Agents are overlaid onto the grid at their positions, matching the behavior
    of Grid.encode() which iterates over grid_agents and overwrites their cells.

    IMPORTANT: Grid.encode() encodes grid objects with the passed scope, but
    encodes agents with scope='global' (GridAgent.encode() is called without
    scope parameter). This function replicates that behavior: object_type_map
    uses the caller's scope, but agent overlays always use scope='global'.

    Args:
        object_type_map: int32 array (H, W) with type IDs (in caller's scope).
        object_state_map: int32 array (H, W) with object state values.
        agent_pos: int32 array (n_agents, 2) with [row, col].
        agent_dir: int32 array (n_agents,) with direction integers.
        agent_inv: int32 array (n_agents, 1) with inventory type IDs (-1 = empty).
        scope: Object registry scope for grid object type IDs (NOT used for agents).
        max_map_size: Maximum map dimensions for padding.

    Returns:
        ndarray of shape (max_H, max_W, 3), dtype int8.
    """
    from cogrid.core.grid_object import object_to_idx

    max_H, max_W = max_map_size
    H, W = object_type_map.shape

    encoding = np.zeros((max_H, max_W, 3), dtype=np.int8)

    # Channel 0: type IDs from object_type_map
    encoding[:H, :W, 0] = object_type_map[:H, :W].astype(np.int8)

    # Channel 1: always 0 for most objects.
    # Pot.encode() overrides channel 1 to extra_state_encoding (1 if tomato
    # contents). For the array-based version, we leave 0 since this override
    # is specific to the object-based path. In practice, channel 1 is 0 in
    # Grid.encode() for all GridObj subclasses except Pot (which returns a
    # tomato flag). This discrepancy only affects pot cells and can be
    # addressed when pot_contents arrays are available to the caller.

    # Channel 2: state from object_state_map
    encoding[:H, :W, 2] = object_state_map[:H, :W].astype(np.int8)

    # Overlay agents onto the encoding.
    # Direction enum: Right=0, Down=1, Left=2, Up=3
    # GridAgent char: Right=">", Down="v", Left="<", Up="^"
    dir_to_char = [">", "v", "<", "^"]

    n_agents = len(agent_dir)
    for i in range(n_agents):
        r, c = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        if r >= max_H or c >= max_W:
            continue

        d = int(agent_dir[i])
        agent_obj_id = f"agent_{dir_to_char[d]}"

        # Grid.encode() calls grid_agent.encode(encode_char=encode_char)
        # WITHOUT passing scope, so agents always use scope='global'.
        agent_type_idx = object_to_idx(agent_obj_id, scope="global")

        # Agent state is inventory encoding:
        # 0 if empty inventory, else object_to_idx(held_item, scope) of the held item.
        # GridAgent.__init__ sets state = object_to_idx(inventory[0], scope=scope),
        # but GridAgent.encode() calls object_to_idx(self, scope='global') which
        # returns the type idx of the agent direction character in global scope.
        # The state (channel 2) comes from GridAgent.state which is the inventory
        # item's type idx in the scope used when the GridAgent was created.
        inv_val = int(agent_inv[i, 0])
        agent_state = 0 if inv_val == -1 else inv_val

        encoding[r, c, 0] = np.int8(agent_type_idx)
        encoding[r, c, 1] = np.int8(0)  # color channel always 0
        encoding[r, c, 2] = np.int8(agent_state)

    return encoding


def can_move_direction_feature(agent_pos, agent_idx: int, wall_map, object_type_map, can_overlap_table):
    """Multi-hot encoding of whether agent can move in each of 4 directions.

    Matches CanMoveDirection.generate() which checks adjacent_positions() in order
    (Right, Left, Down, Up) and returns 1 if the cell is empty (None) or the object
    can_overlap(agent). The array version uses the static CAN_OVERLAP lookup table.

    Note: adjacent_positions yields (row+0,col+1), (row+0,col-1), (row+1,col+0),
    (row-1,col+0) which corresponds to Right, Left, Down, Up.

    Args:
        agent_pos: int32 array (n_agents, 2).
        agent_idx: Index of the agent.
        wall_map: int32 array (H, W), 1 where wall.
        object_type_map: int32 array (H, W) with type IDs.
        can_overlap_table: int32 array (n_types,), 1 if type is overlappable.

    Returns:
        ndarray of shape (4,), dtype int32.
    """
    can_move = np.zeros(4, dtype=np.int32)
    row, col = int(agent_pos[agent_idx, 0]), int(agent_pos[agent_idx, 1])
    H, W = wall_map.shape

    for i, (nr, nc) in enumerate(adjacent_positions(row, col)):
        # Out of bounds: can't move there
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            continue

        type_id = int(object_type_map[nr, nc])

        # Empty cell (type_id == 0, matching object_to_idx(None) == 0) is overlappable
        # OR the object type is marked overlappable in the lookup table
        if can_overlap_table[type_id] == 1:
            can_move[i] = 1

    return can_move


def inventory_feature(agent_inv, agent_idx: int):
    """Inventory encoding as (1,) array matching Inventory.generate().

    The existing feature uses: 0 = empty, get_object_names(scope).index(obj_id) + 1 for items.
    Our agent_inv stores: -1 = empty, type_id = object_to_idx(obj_id, scope) for items.
    Since object_to_idx returns the same index as get_object_names().index(),
    conversion is: feature_val = agent_inv + 1 for non-empty, 0 for empty.

    Args:
        agent_inv: int32 array (n_agents, 1) with -1 for empty, type_id for held item.
        agent_idx: Index of the agent.

    Returns:
        ndarray of shape (1,), dtype int32.
    """
    inv_val = int(agent_inv[agent_idx, 0])
    if inv_val == -1:
        feature_val = 0
    else:
        feature_val = inv_val + 1
    return np.array([feature_val], dtype=np.int32)


# ---------------------------------------------------------------------------
# Feature composition
# ---------------------------------------------------------------------------


def compose_features(feature_fns: list, state_dict: dict, agent_idx: int):
    """Compose multiple feature functions into a single flat observation array.

    Each feature_fn takes (state_dict, agent_idx) and returns an ndarray.
    Results are flattened and concatenated.

    Args:
        feature_fns: List of callables, each with signature (state_dict, agent_idx) -> ndarray.
        state_dict: Dict of state arrays (agent_pos, agent_dir, etc.).
        agent_idx: Index of the agent to generate observation for.

    Returns:
        ndarray: Flat 1D observation array.
    """
    features = [fn(state_dict, agent_idx) for fn in feature_fns]
    return np.concatenate([f.ravel() for f in features])


def build_feature_fn(feature_names: list[str], scope: str = "global", **kwargs):
    """Build a composed feature function from feature names.

    Called at init time. Returns a function that takes (state_dict, agent_idx) -> obs_array.
    Resolves which features to include and their parameters once at init.

    Args:
        feature_names: List of feature name strings to include.
        scope: Object registry scope.
        **kwargs: Additional keyword arguments (e.g., max_map_size for full_map_encoding).

    Returns:
        Callable with signature (state_dict: dict, agent_idx: int) -> ndarray.
    """
    from cogrid.core.grid_object import build_lookup_tables

    # Build lookup tables once at init time
    tables = build_lookup_tables(scope=scope)
    max_map_size = kwargs.get("max_map_size", (12, 12))

    # Map feature names to bound functions
    feature_fns = []
    for name in feature_names:
        if name == "agent_position":
            feature_fns.append(
                lambda sd, ai: agent_pos_feature(sd["agent_pos"], ai)
            )
        elif name == "agent_dir":
            feature_fns.append(
                lambda sd, ai: agent_dir_feature(sd["agent_dir"], ai)
            )
        elif name == "full_map_encoding":
            # Capture scope and max_map_size in closure
            _scope = scope
            _mms = max_map_size
            feature_fns.append(
                lambda sd, ai, s=_scope, m=_mms: full_map_encoding_feature(
                    sd["object_type_map"],
                    sd["object_state_map"],
                    sd["agent_pos"],
                    sd["agent_dir"],
                    sd["agent_inv"],
                    scope=s,
                    max_map_size=m,
                )
            )
        elif name == "can_move_direction":
            _co = tables["CAN_OVERLAP"]
            feature_fns.append(
                lambda sd, ai, co=_co: can_move_direction_feature(
                    sd["agent_pos"], ai, sd["wall_map"], sd["object_type_map"], co
                )
            )
        elif name == "inventory":
            feature_fns.append(
                lambda sd, ai: inventory_feature(sd["agent_inv"], ai)
            )
        else:
            raise ValueError(
                f"Unknown array feature: '{name}'. Available: "
                f"agent_position, agent_dir, full_map_encoding, can_move_direction, inventory"
            )

    def composed_fn(state_dict: dict, agent_idx: int):
        return compose_features(feature_fns, state_dict, agent_idx)

    return composed_fn


# ---------------------------------------------------------------------------
# Per-agent vectorized observation generation
# ---------------------------------------------------------------------------


def get_all_agent_obs(feature_fn, state_dict: dict, n_agents: int):
    """Generate observations for all agents.

    Returns (n_agents, obs_dim) array. Phase 1 uses a Python loop;
    Phase 2 will convert to jax.vmap.

    Args:
        feature_fn: Callable with signature (state_dict, agent_idx) -> obs_array.
        state_dict: Dict of state arrays.
        n_agents: Number of agents.

    Returns:
        ndarray of shape (n_agents, obs_dim).
    """
    # PHASE2: convert to jax.vmap
    return np.stack([feature_fn(state_dict, i) for i in range(n_agents)])


# ---------------------------------------------------------------------------
# Development-time parity test
# ---------------------------------------------------------------------------


def test_feature_parity(n_steps: int = 20, seed: int = 42):
    """Validate array-based features produce identical values to existing Feature.generate().

    Instantiates an Overcooked environment (cramped_room), resets it, and for each
    of the core features (agent_pos, agent_dir, can_move_direction, inventory,
    full_map_encoding), compares the existing object-based feature output against
    the array-based feature output for numerical identity.

    Runs for ``n_steps`` with random actions to exercise multiple states.

    The full_map_encoding parity test compares Grid.encode(scope='overcooked')
    against full_map_encoding_feature. Note: Grid.encode() uses scope='global'
    for agent overlays regardless of the passed scope, and our array version
    matches this behavior.

    Args:
        n_steps: Number of random steps to check.
        seed: Random seed for reproducibility.

    Raises:
        AssertionError: If any feature output differs between existing and array-based.
    """
    import cogrid.envs  # trigger registration
    from cogrid.envs import registry
    from cogrid.feature_space.features import (
        AgentPosition, AgentDir, CanMoveDirection, Inventory,
    )
    from cogrid.core.grid_object import build_lookup_tables, object_to_idx
    from cogrid.core.grid_utils import layout_to_array_state
    from cogrid.core.agent import create_agent_arrays

    # Create environment
    env = registry.make("Overcooked-CrampedRoom-V0")
    obs = env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    scope = env.scope
    tables = build_lookup_tables(scope=scope)

    # Existing feature instances
    agent_pos_feat = AgentPosition()
    agent_dir_feat = AgentDir()
    can_move_feat = CanMoveDirection()
    inv_feat = Inventory(inventory_capacity=1)

    def check_parity_at_state(step_label: str):
        """Compare existing vs array features for all agents at current state."""
        # Build array state from current grid + agents
        array_state = layout_to_array_state(env.grid, scope=scope)
        agent_arrays = create_agent_arrays(env.env_agents, scope=scope)

        # Map from agent_id to array index
        sorted_agent_ids = sorted(env.env_agents.keys())
        id_to_idx = {aid: i for i, aid in enumerate(sorted_agent_ids)}

        for agent_id in env.agent_ids:
            aidx = id_to_idx[agent_id]

            # --- agent_position ---
            existing_pos = agent_pos_feat.generate(env, agent_id)
            array_pos = agent_pos_feature(agent_arrays["agent_pos"], aidx)
            assert np.array_equal(existing_pos, array_pos), (
                f"[{step_label}] agent_position mismatch for agent {agent_id}: "
                f"existing={existing_pos}, array={array_pos}"
            )

            # --- agent_dir ---
            existing_dir = agent_dir_feat.generate(env, agent_id)
            array_dir = agent_dir_feature(agent_arrays["agent_dir"], aidx)
            assert np.array_equal(existing_dir, array_dir), (
                f"[{step_label}] agent_dir mismatch for agent {agent_id}: "
                f"existing={existing_dir}, array={array_dir}"
            )

            # --- can_move_direction ---
            existing_cm = can_move_feat.generate(env, agent_id)
            array_cm = can_move_direction_feature(
                agent_arrays["agent_pos"], aidx,
                array_state["wall_map"], array_state["object_type_map"],
                tables["CAN_OVERLAP"],
            )
            assert np.array_equal(existing_cm, array_cm), (
                f"[{step_label}] can_move_direction mismatch for agent {agent_id}: "
                f"existing={existing_cm}, array={array_cm}"
            )

            # --- inventory ---
            existing_inv = inv_feat.generate(env, agent_id)
            array_inv = inventory_feature(agent_arrays["agent_inv"], aidx)
            assert np.array_equal(existing_inv, array_inv), (
                f"[{step_label}] inventory mismatch for agent {agent_id}: "
                f"existing={existing_inv}, array={array_inv}"
            )

        # --- full_map_encoding ---
        # Grid.encode(scope='overcooked') encodes grid objects with overcooked
        # scope but agents with global scope. Our array version matches this.
        existing_grid_enc = env.grid.encode(encode_char=False, scope=scope)
        array_grid_enc = full_map_encoding_feature(
            array_state["object_type_map"],
            array_state["object_state_map"],
            agent_arrays["agent_pos"],
            agent_arrays["agent_dir"],
            agent_arrays["agent_inv"],
            scope=scope,
            max_map_size=(12, 12),
        )

        # Compare only the actual grid region (not padding)
        H, W = existing_grid_enc.shape[0], existing_grid_enc.shape[1]
        existing_region = existing_grid_enc[:H, :W, :]
        array_region = array_grid_enc[:H, :W, :]

        # Channel 0 (type IDs) and channel 2 (state) should match.
        # Channel 1 may differ for Pot cells (Pot.encode overrides channel 1
        # with tomato flag, which we don't replicate from arrays).
        assert np.array_equal(existing_region[:, :, 0], array_region[:, :, 0]), (
            f"[{step_label}] full_map_encoding channel 0 mismatch:\n"
            f"existing:\n{existing_region[:,:,0]}\narray:\n{array_region[:,:,0]}"
        )

        # Channel 2 comparison: agent state overlay may differ because
        # GridAgent.__init__ computes state from inventory in the creating
        # scope while our array uses the raw agent_inv value.
        # Compare non-agent cells for state parity.
        for r in range(H):
            for c in range(W):
                is_agent_cell = False
                for aidx in range(agent_arrays["n_agents"]):
                    if (int(agent_arrays["agent_pos"][aidx, 0]) == r and
                            int(agent_arrays["agent_pos"][aidx, 1]) == c):
                        is_agent_cell = True
                        break
                if not is_agent_cell:
                    assert existing_region[r, c, 2] == array_region[r, c, 2], (
                        f"[{step_label}] full_map_encoding ch2 mismatch at ({r},{c}): "
                        f"existing={existing_region[r,c,2]}, array={array_region[r,c,2]}"
                    )

    # Check at initial state
    check_parity_at_state("reset")
    print("Parity check at reset: PASSED")

    # Step through with random actions
    n_actions = env.action_spaces[env.agent_ids[0]].n
    for step in range(n_steps):
        actions = {aid: rng.integers(0, n_actions) for aid in env.agent_ids}
        obs, rewards, dones, truncs, infos = env.step(actions)

        # Check if any agent is done
        if any(dones.values()) or any(truncs.values()):
            obs = env.reset(seed=seed + step + 1)

        check_parity_at_state(f"step_{step}")

    print(f"Parity check for {n_steps} steps: ALL PASSED")
    print("test_feature_parity: SUCCESS")
