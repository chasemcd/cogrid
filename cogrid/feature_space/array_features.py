"""Array-based feature extractors operating on state arrays instead of Grid/Agent objects.

These functions are standalone alternatives to the Feature.generate() class methods.
They operate directly on state arrays (agent_pos, agent_dir, object_type_map, etc.)
produced by layout_to_array_state() and create_agent_arrays(), producing numerically
identical observations to the existing feature system.

Feature composition is handled by autowire via ``compose_feature_fns()`` in
``cogrid/core/array_features.py``. Each ArrayFeature subclass provides a
``build_feature_fn(cls, scope)`` classmethod that returns a closure.

All functions use ``xp`` (the backend-agnostic array namespace) so they work
identically on both numpy and JAX backends. No ``_jax`` variants exist; a single
implementation serves both paths.
"""

from __future__ import annotations

from cogrid.backend import xp
from cogrid.core.array_features import ArrayFeature, register_feature_type


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
# ArrayFeature subclasses
# ---------------------------------------------------------------------------


@register_feature_type("agent_dir", scope="global")
class AgentDir(ArrayFeature):
    per_agent = True
    obs_dim = 4

    @classmethod
    def build_feature_fn(cls, scope):
        def fn(state_dict, agent_idx):
            return agent_dir_feature(state_dict.agent_dir, agent_idx)
        return fn


@register_feature_type("agent_position", scope="global")
class AgentPosition(ArrayFeature):
    per_agent = True
    obs_dim = 2

    @classmethod
    def build_feature_fn(cls, scope):
        def fn(state_dict, agent_idx):
            return agent_pos_feature(state_dict.agent_pos, agent_idx)
        return fn


@register_feature_type("can_move_direction", scope="global")
class CanMoveDirection(ArrayFeature):
    per_agent = True
    obs_dim = 4

    @classmethod
    def build_feature_fn(cls, scope):
        from cogrid.core.grid_object import build_lookup_tables
        tables = build_lookup_tables(scope=scope)
        can_overlap_table = xp.array(tables["CAN_OVERLAP"], dtype=xp.int32)

        def fn(state_dict, agent_idx):
            return can_move_direction_feature(
                state_dict.agent_pos,
                agent_idx,
                state_dict.wall_map,
                state_dict.object_type_map,
                can_overlap_table,
            )
        return fn


@register_feature_type("inventory", scope="global")
class Inventory(ArrayFeature):
    per_agent = True
    obs_dim = 1

    @classmethod
    def build_feature_fn(cls, scope):
        def fn(state_dict, agent_idx):
            return inventory_feature(state_dict.agent_inv, agent_idx)
        return fn


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
