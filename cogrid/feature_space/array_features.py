"""Array-based feature extractors operating on state arrays instead of Grid/Agent objects.

These functions are standalone alternatives to the Feature.generate() class methods.
They operate directly on state arrays (agent_pos, agent_dir, object_type_map, etc.)
produced by layout_to_array_state() and create_agent_arrays(), producing numerically
identical observations to the existing feature system.

Feature composition (which features to include, in what order) is resolved at init
time via build_feature_fn(), producing a single composed function that can be called
with state arrays.

All functions use ``xp`` (the backend-agnostic array namespace) so they work
identically on both numpy and JAX backends. No ``_jax`` variants exist; a single
implementation serves both paths.
"""

from __future__ import annotations

from cogrid.backend import xp


# ---------------------------------------------------------------------------
# Core feature extractors
# ---------------------------------------------------------------------------


def agent_pos_feature(agent_pos, agent_idx):
    """Extract agent position as (2,) int32 array.

    Args:
        agent_pos: int32 array of shape (n_agents, 2) with [row, col].
        agent_idx: Index of the agent to extract position for.

    Returns:
        ndarray of shape (2,), dtype int32.
    """
    return agent_pos[agent_idx].astype(xp.int32)


def agent_dir_feature(agent_dir, agent_idx):
    """One-hot encoding of agent direction as (4,) int32 array.

    Args:
        agent_dir: int32 array of shape (n_agents,) with direction integers.
        agent_idx: Index of the agent.

    Returns:
        ndarray of shape (4,), dtype int32 with exactly one 1.
    """
    return (xp.arange(4) == agent_dir[agent_idx]).astype(xp.int32)


def full_map_encoding_feature(
    object_type_map,
    object_state_map,
    agent_pos,
    agent_dir,
    agent_inv,
    agent_type_ids,
    max_map_size=(12, 12),
):
    """Full map encoding as (max_H, max_W, 3) int8 array.

    Produces the same 3-channel encoding as FullMapEncoding.generate():
    - Channel 0: type IDs (with agent overlays)
    - Channel 1: extra state encoding (always 0 for now)
    - Channel 2: object state values (with agent inventory state overlay)

    Agents are overlaid onto the grid at their positions using pre-computed
    ``agent_type_ids`` (resolved at init time by build_feature_fn).

    Args:
        object_type_map: int32 array (H, W) with type IDs.
        object_state_map: int32 array (H, W) with object state values.
        agent_pos: int32 array (n_agents, 2) with [row, col].
        agent_dir: int32 array (n_agents,) with direction integers.
        agent_inv: int32 array (n_agents, 1) with inventory type IDs (-1 = empty).
        agent_type_ids: int32 array (4,) where agent_type_ids[dir] gives the
            global-scope type_id for the agent character facing that direction.
            Pre-computed at init time from object_to_idx.
        max_map_size: Maximum map dimensions for padding.

    Returns:
        ndarray of shape (max_H, max_W, 3), dtype int8.
    """
    from cogrid.backend.array_ops import set_at_2d

    max_H, max_W = max_map_size
    H, W = object_type_map.shape

    # Build channels using xp.pad to avoid backend-specific slice assignment
    pad_h = max_H - H
    pad_w = max_W - W
    ch0 = xp.pad(object_type_map.astype(xp.int8), ((0, pad_h), (0, pad_w)))
    ch1 = xp.zeros((max_H, max_W), dtype=xp.int8)
    ch2 = xp.pad(object_state_map.astype(xp.int8), ((0, pad_h), (0, pad_w)))

    # Agent overlay: type_id per agent based on direction
    agent_type = agent_type_ids[agent_dir]  # (n_agents,)
    agent_state = xp.where(agent_inv[:, 0] == -1, 0, agent_inv[:, 0])  # (n_agents,)

    rows = agent_pos[:, 0]
    cols = agent_pos[:, 1]

    # Scatter agents onto channels using set_at_2d (loop over n_agents which is
    # static/tiny, typically 2). Uses array-valued indices, no int() casts.
    for i_agent in range(agent_pos.shape[0]):
        r, c = rows[i_agent], cols[i_agent]
        ch0 = set_at_2d(ch0, r, c, agent_type[i_agent].astype(xp.int8))
        ch1 = set_at_2d(ch1, r, c, xp.int8(0))
        ch2 = set_at_2d(ch2, r, c, agent_state[i_agent].astype(xp.int8))

    encoding = xp.stack([ch0, ch1, ch2], axis=-1)
    return encoding


def can_move_direction_feature(agent_pos, agent_idx, wall_map, object_type_map, can_overlap_table):
    """Multi-hot encoding of whether agent can move in each of 4 directions.

    Direction order matches adjacent_positions: Right, Left, Down, Up.
    Uses vectorized deltas array instead of Python loop.

    Args:
        agent_pos: int32 array (n_agents, 2).
        agent_idx: Index of the agent.
        wall_map: int32 array (H, W), 1 where wall.
        object_type_map: int32 array (H, W) with type IDs.
        can_overlap_table: int32 array (n_types,), 1 if type is overlappable.

    Returns:
        ndarray of shape (4,), dtype int32.
    """
    H, W = wall_map.shape

    # 4 directions matching adjacent_positions order: Right, Left, Down, Up
    deltas = xp.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=xp.int32)
    pos = agent_pos[agent_idx]  # (2,)
    neighbors = pos[None, :] + deltas  # (4, 2)

    # Check bounds
    in_bounds = (
        (neighbors[:, 0] >= 0)
        & (neighbors[:, 0] < H)
        & (neighbors[:, 1] >= 0)
        & (neighbors[:, 1] < W)
    )

    # Clip to valid indices for safe array access (out-of-bounds masked by in_bounds)
    clipped = xp.clip(neighbors, xp.array([0, 0]), xp.array([H - 1, W - 1]))
    type_ids = object_type_map[clipped[:, 0], clipped[:, 1]]  # (4,)
    can_overlap = can_overlap_table[type_ids]  # (4,)

    return (in_bounds & (can_overlap == 1)).astype(xp.int32)


def inventory_feature(agent_inv, agent_idx):
    """Inventory encoding as (1,) array. 0 = empty, type_id+1 otherwise.

    Args:
        agent_inv: int32 array (n_agents, 1) with -1 for empty, type_id for held item.
        agent_idx: Index of the agent.

    Returns:
        ndarray of shape (1,), dtype int32.
    """
    inv_val = agent_inv[agent_idx, 0]
    feature_val = xp.where(inv_val == -1, 0, inv_val + 1)
    return xp.array([feature_val], dtype=xp.int32)


# ---------------------------------------------------------------------------
# Feature composition
# ---------------------------------------------------------------------------


def compose_features(feature_fns, state_dict, agent_idx):
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
    return xp.concatenate([f.ravel() for f in features])


def build_feature_fn(feature_names, scope="global", **kwargs):
    """Build a composed feature function from feature names. Works on both backends.

    Called at init time. Returns a function that takes (state_dict, agent_idx) -> obs_array.
    Resolves which features to include and pre-computes lookup tables once.

    Args:
        feature_names: List of feature name strings to include.
        scope: Object registry scope.
        **kwargs: Additional keyword arguments (e.g., max_map_size for full_map_encoding).

    Returns:
        Callable with signature (state_dict: dict, agent_idx: int) -> ndarray.
    """
    from cogrid.core.grid_object import build_lookup_tables, object_to_idx

    tables = build_lookup_tables(scope=scope)
    max_map_size = kwargs.get("max_map_size", (12, 12))

    # Pre-compute agent_type_ids for full_map_encoding
    dir_to_char = [">", "v", "<", "^"]
    agent_type_ids = xp.array(
        [object_to_idx(f"agent_{c}", scope="global") for c in dir_to_char],
        dtype=xp.int32,
    )

    can_overlap_table = xp.array(tables["CAN_OVERLAP"], dtype=xp.int32)

    feature_fns = []
    for name in feature_names:
        if name == "agent_position":
            feature_fns.append(lambda sd, ai: agent_pos_feature(sd["agent_pos"], ai))
        elif name == "agent_dir":
            feature_fns.append(lambda sd, ai: agent_dir_feature(sd["agent_dir"], ai))
        elif name == "full_map_encoding":
            _atids = agent_type_ids
            _mms = max_map_size
            feature_fns.append(
                lambda sd, ai, atids=_atids, mms=_mms: full_map_encoding_feature(
                    sd["object_type_map"],
                    sd["object_state_map"],
                    sd["agent_pos"],
                    sd["agent_dir"],
                    sd["agent_inv"],
                    agent_type_ids=atids,
                    max_map_size=mms,
                )
            )
        elif name == "can_move_direction":
            _co = can_overlap_table
            feature_fns.append(
                lambda sd, ai, co=_co: can_move_direction_feature(
                    sd["agent_pos"], ai, sd["wall_map"], sd["object_type_map"], co
                )
            )
        elif name == "inventory":
            feature_fns.append(lambda sd, ai: inventory_feature(sd["agent_inv"], ai))
        else:
            raise ValueError(f"Unknown array feature: '{name}'")

    def composed_fn(state_dict, agent_idx):
        return compose_features(feature_fns, state_dict, agent_idx)

    return composed_fn


# ---------------------------------------------------------------------------
# Per-agent vectorized observation generation
# ---------------------------------------------------------------------------


def get_all_agent_obs(feature_fn, state_dict, n_agents):
    """Generate observations for all agents.

    Returns (n_agents, obs_dim) array. Uses Python loop with xp.stack.
    vmap optimization deferred to Phase 8.

    Args:
        feature_fn: Callable with signature (state_dict, agent_idx) -> obs_array.
        state_dict: Dict of state arrays.
        n_agents: Number of agents.

    Returns:
        ndarray of shape (n_agents, obs_dim).
    """
    return xp.stack([feature_fn(state_dict, i) for i in range(n_agents)])
