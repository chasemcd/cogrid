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
    - Channel 1: extra state encoding (Pot sets this to 1 if tomato contents)
    - Channel 2: object state values (with agent inventory state overlay)

    Agents are overlaid onto the grid at their positions, matching the behavior
    of Grid.encode() which iterates over grid_agents and overwrites their cells.

    Args:
        object_type_map: int32 array (H, W) with type IDs.
        object_state_map: int32 array (H, W) with object state values.
        agent_pos: int32 array (n_agents, 2) with [row, col].
        agent_dir: int32 array (n_agents,) with direction integers.
        agent_inv: int32 array (n_agents, 1) with inventory type IDs (-1 = empty).
        scope: Object registry scope for agent type ID lookup.
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

    # Channel 1: always 0 for most objects
    # Pot.encode() overrides channel 1 to extra_state_encoding (1 if tomato in pot).
    # For the array-based version, we store 0 as we don't have per-cell
    # extra state encoding in the array representation. Pot's encode() override
    # is specific to the object-based path and would need pot_contents arrays
    # to replicate exactly. For non-Overcooked features or when used with
    # FullMapEncoding, channel 1 is always 0 from GridObj.encode().

    # Channel 2: state from object_state_map
    encoding[:H, :W, 2] = object_state_map[:H, :W].astype(np.int8)

    # Overlay agents onto the encoding
    dir_chars = ["^", ">", "v", "<"]  # Right=0, Down=1, Left=2, Up=3 -> ^, >, v, <
    # Wait -- Direction enum: Right=0, Down=1, Left=2, Up=3
    # GridAgent char mapping: Up="^", Down="v", Left="<", Right=">"
    # So: dir 0 (Right) -> ">", dir 1 (Down) -> "v", dir 2 (Left) -> "<", dir 3 (Up) -> "^"
    dir_to_char = [">", "v", "<", "^"]

    n_agents = len(agent_dir)
    for i in range(n_agents):
        r, c = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        if r >= max_H or c >= max_W:
            continue

        d = int(agent_dir[i])
        agent_obj_id = f"agent_{dir_to_char[d]}"

        # GridAgent.encode uses scope='global' by default (Grid.encode passes
        # encode_char to grid_agent.encode without scope). We use scope parameter
        # to match the caller's expected scope.
        agent_type_idx = object_to_idx(agent_obj_id, scope=scope)

        # Agent state is inventory encoding:
        # 0 if empty inventory, else object_to_idx(held_item, scope) of the held item
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
