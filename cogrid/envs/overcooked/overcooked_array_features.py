"""Array-based Overcooked feature extractors.

Produces the same 677-dim (for 2 agents) ego-centric observation as
``OvercookedCollectedFeatures.generate()``, but operates entirely on
state arrays so it works with both numpy and JAX backends.

All functions are JIT-compatible: no Python control flow on traced
values, no int()/np.array() conversions of traced arrays.

Public API:
    ``build_overcooked_feature_fn`` -- returned by ``Pot.build_feature_fn()``.
"""

from __future__ import annotations

from cogrid.core.array_features import ArrayFeature, register_feature_type


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def overcooked_inventory_feature(agent_inv, agent_idx, inv_type_ids):
    """One-hot inventory encoding. (5,) = [onion, onion_soup, plate, tomato, tomato_soup].

    Empty inventory (-1) matches nothing -> all zeros.
    """
    from cogrid.backend import xp

    held = agent_inv[agent_idx, 0]
    return (inv_type_ids == held).astype(xp.int32)


def next_to_counter_feature(agent_pos, agent_idx, object_type_map, counter_type_id):
    """Multi-hot cardinal adjacency to counters. (4,) = [R, L, D, U]."""
    from cogrid.backend import xp

    H, W = object_type_map.shape
    deltas = xp.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=xp.int32)
    pos = agent_pos[agent_idx]
    neighbors = pos[None, :] + deltas

    in_bounds = (
        (neighbors[:, 0] >= 0) & (neighbors[:, 0] < H)
        & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < W)
    )

    clipped = xp.clip(neighbors, xp.array([0, 0]), xp.array([H - 1, W - 1]))
    types = object_type_map[clipped[:, 0], clipped[:, 1]]

    return (in_bounds & (types == counter_type_id)).astype(xp.int32)


def next_to_pot_feature(
    agent_pos, agent_idx, object_type_map, pot_type_id,
    pot_positions, pot_contents, pot_timer, capacity=3,
):
    """(16,) adjacency to pots, with status encoding.

    Layout: [empty(4), partial(4), cooking(4), ready(4)], indexed by direction.
    Fully vectorized over directions and pots — no Python control flow on traced values.
    """
    from cogrid.backend import xp

    H, W = object_type_map.shape
    n_pots = pot_positions.shape[0]
    deltas = xp.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=xp.int32)
    pos = agent_pos[agent_idx]
    neighbors = pos[None, :] + deltas  # (4, 2)

    in_bounds = (
        (neighbors[:, 0] >= 0) & (neighbors[:, 0] < H)
        & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < W)
    )

    clipped = xp.clip(neighbors, xp.array([0, 0]), xp.array([H - 1, W - 1]))
    types = object_type_map[clipped[:, 0], clipped[:, 1]]
    is_pot = in_bounds & (types == pot_type_id)  # (4,)

    # For each direction, find which pot it matches (if any)
    # neighbor_rc: (4, 1, 2) vs pot_positions: (1, n_pots, 2)
    neighbor_rc = clipped[:, None, :]  # (4, 1, 2)
    pot_rc = pot_positions[None, :, :]  # (1, n_pots, 2)
    match = (neighbor_rc[:, :, 0] == pot_rc[:, :, 0]) & (neighbor_rc[:, :, 1] == pot_rc[:, :, 1])  # (4, n_pots)

    # Count items per pot and get timer: (n_pots,)
    items_per_pot = xp.sum(pot_contents > 0, axis=1)  # (n_pots,)

    # Pot status per pot: [empty, partial, cooking, ready]
    is_empty = (items_per_pot == 0)          # (n_pots,)
    is_partial = (items_per_pot > 0) & (items_per_pot < capacity)  # (n_pots,)
    is_cooking = (items_per_pot == capacity) & (pot_timer > 0)     # (n_pots,)
    is_ready = (items_per_pot == capacity) & (pot_timer == 0)      # (n_pots,)

    # For each direction, check if any matched pot has each status
    # match: (4, n_pots), is_empty: (n_pots,) -> broadcast to (4, n_pots)
    dir_empty = xp.any(match & is_empty[None, :], axis=1)    # (4,)
    dir_partial = xp.any(match & is_partial[None, :], axis=1)  # (4,)
    dir_cooking = xp.any(match & is_cooking[None, :], axis=1)  # (4,)
    dir_ready = xp.any(match & is_ready[None, :], axis=1)     # (4,)

    # Mask by is_pot
    result = xp.concatenate([
        (is_pot & dir_empty).astype(xp.int32),
        (is_pot & dir_partial).astype(xp.int32),
        (is_pot & dir_cooking).astype(xp.int32),
        (is_pot & dir_ready).astype(xp.int32),
    ])

    return result


def closest_obj_feature(agent_pos, agent_idx, object_type_map, object_state_map,
                         target_type_id, n, large_dist=9999):
    """(2*n,) deltas (dy, dx) to the n closest instances of target_type_id.

    Searches both object_type_map (for placed objects like pots/stacks) and
    object_state_map (for items placed on counters).
    Fully vectorized — uses argsort with masked distances.
    """
    from cogrid.backend import xp

    H, W = object_type_map.shape
    pos = agent_pos[agent_idx]  # (2,)

    type_flat = object_type_map.ravel()   # (H*W,)
    state_flat = object_state_map.ravel()  # (H*W,)
    match = (type_flat == target_type_id) | (state_flat == target_type_id)

    # Coordinate arrays
    rows = xp.arange(H, dtype=xp.int32).repeat(W)
    cols = xp.tile(xp.arange(W, dtype=xp.int32), H)

    # Exclude agent's own position
    agent_mask = ~((rows == pos[0]) & (cols == pos[1]))
    match = match & agent_mask

    # Deltas and manhattan distances
    dys = pos[0] - rows   # (H*W,)
    dxs = pos[1] - cols   # (H*W,)
    dists = xp.abs(dys) + xp.abs(dxs)  # (H*W,)

    # Mask non-matches with large distance so they sort last
    masked_dists = xp.where(match, dists, large_dist)

    # Sort and take top-n
    sorted_indices = xp.argsort(masked_dists)
    top_indices = sorted_indices[:n]  # (n,)

    # Gather deltas; zero out entries where there was no match
    top_dys = dys[top_indices]
    top_dxs = dxs[top_indices]
    top_valid = masked_dists[top_indices] < large_dist

    top_dys = xp.where(top_valid, top_dys, 0)
    top_dxs = xp.where(top_valid, top_dxs, 0)

    # Interleave (dy0, dx0, dy1, dx1, ...)
    result = xp.zeros(2 * n, dtype=xp.int32)
    for i in range(n):
        result = result.at[2 * i].set(top_dys[i]) if hasattr(result, 'at') else _set_idx(result, 2 * i, top_dys[i])
        result = result.at[2 * i + 1].set(top_dxs[i]) if hasattr(result, 'at') else _set_idx(result, 2 * i + 1, top_dxs[i])

    return result


def ordered_pot_features(
    agent_pos, agent_idx, pot_positions, pot_contents, pot_timer,
    max_num_pots, onion_id, tomato_id, capacity=3,
):
    """(12 * max_num_pots,) per-pot features in grid-scan order.

    Per pot: [reachable(1), status(4), contents(2), timer(1), distance(2), location(2)] = 12.
    Status one-hot: [ready, empty, full/cooking, partial] matching _calc_pot_features order.
    Fully vectorized — no Python control flow on traced values.
    """
    from cogrid.backend import xp

    pos = agent_pos[agent_idx]  # (2,)

    # Items per pot: (n_pots,)
    n_items = xp.sum(pot_contents > 0, axis=1)

    # Status one-hot per pot: [ready, empty, full/cooking, partial]
    is_ready = ((n_items == capacity) & (pot_timer == 0)).astype(xp.float32)
    is_empty = (n_items == 0).astype(xp.float32)
    is_cooking = ((n_items == capacity) & (pot_timer > 0)).astype(xp.float32)
    is_partial = ((n_items > 0) & (n_items < capacity)).astype(xp.float32)

    # Contents per pot: count of onion and tomato
    n_onion = xp.sum(pot_contents == onion_id, axis=1).astype(xp.float32)   # (n_pots,)
    n_tomato = xp.sum(pot_contents == tomato_id, axis=1).astype(xp.float32)  # (n_pots,)

    # Timer: value if cooking, -1 if not
    is_cooking_bool = (n_items == capacity) & (pot_timer > 0)
    timer_val = xp.where(is_cooking_bool, pot_timer.astype(xp.float32), xp.float32(-1))

    # Distance and location
    dy = (pos[0] - pot_positions[:, 0]).astype(xp.float32)  # (n_pots,)
    dx = (pos[1] - pot_positions[:, 1]).astype(xp.float32)  # (n_pots,)
    pr = pot_positions[:, 0].astype(xp.float32)
    pc = pot_positions[:, 1].astype(xp.float32)

    # Reachable = 1 for all pots (matching Python impl)
    reachable = xp.ones(pot_positions.shape[0], dtype=xp.float32)

    # Per-pot feature vector: (n_pots, 12)
    per_pot = xp.stack([
        reachable, is_ready, is_empty, is_cooking, is_partial,
        n_onion, n_tomato, timer_val, dy, dx, pr, pc,
    ], axis=1)  # (n_pots, 12)

    # Pad to max_num_pots (if fewer pots than max)
    n_pots = pot_positions.shape[0]
    if n_pots < max_num_pots:
        pad = xp.zeros((max_num_pots - n_pots, 12), dtype=xp.float32)
        per_pot = xp.concatenate([per_pot, pad], axis=0)

    # Take only max_num_pots and flatten
    return per_pot[:max_num_pots].ravel()


def dist_to_other_players_feature(agent_pos, agent_idx, n_agents):
    """(2 * (n_agents - 1),) distance to other agents."""
    from cogrid.backend import xp

    pos = agent_pos[agent_idx]
    result = xp.zeros(2 * (n_agents - 1), dtype=xp.int32)
    out_idx = 0
    for i in range(n_agents):
        if i == agent_idx:  # agent_idx is a Python int, this is fine under JIT
            continue
        dy = pos[0] - agent_pos[i, 0]
        dx = pos[1] - agent_pos[i, 1]
        result = result.at[out_idx].set(dy) if hasattr(result, 'at') else _set_idx(result, out_idx, dy)
        result = result.at[out_idx + 1].set(dx) if hasattr(result, 'at') else _set_idx(result, out_idx + 1, dx)
        out_idx += 2
    return result


def layout_id_feature(layout_idx, num_layouts=5):
    """(num_layouts,) one-hot layout identifier."""
    from cogrid.backend import xp

    return (xp.arange(num_layouts) == layout_idx).astype(xp.int32)


def environment_layout_feature(object_type_map, layout_type_ids, max_shape):
    """(6 * max_shape[0] * max_shape[1],) binary masks for each of 6 object types.

    Types: [counter, pot, onion, plate, onion_stack, plate_stack].
    Uses the same (max_shape[0], max_shape[1]) convention and flat indexing
    as the Python ``EnvironmentLayout`` feature: ``flat_index = row * max_shape[1] + col``.
    Vectorized using scatter-style indexing (no Python loops on traced values).
    """
    from cogrid.backend import xp

    H, W = object_type_map.shape
    dim0, dim1 = max_shape
    channel_size = dim0 * dim1
    n_types = len(layout_type_ids)

    # Build flat coordinate indices: flat_index[r, c] = r * dim1 + c
    row_idx = xp.arange(H, dtype=xp.int32)[:, None] * dim1  # (H, 1)
    col_idx = xp.arange(W, dtype=xp.int32)[None, :]         # (1, W)
    flat_idx = (row_idx + col_idx).ravel()  # (H*W,)

    otm_flat = object_type_map.ravel()  # (H*W,)

    result = xp.zeros(n_types * channel_size, dtype=xp.int32)

    for t_idx in range(n_types):
        type_id = layout_type_ids[t_idx]
        offset = t_idx * channel_size
        is_match = (otm_flat == type_id).astype(xp.int32)  # (H*W,)
        # Scatter matched positions into the result
        target_indices = flat_idx + offset  # (H*W,)
        if hasattr(result, 'at'):  # JAX
            result = result.at[target_indices].add(is_match)
        else:
            for j in range(len(target_indices)):
                if is_match[j]:
                    result[target_indices[j]] = 1

    return result


# ---------------------------------------------------------------------------
# Helper: backend-agnostic element assignment
# ---------------------------------------------------------------------------


def _set_idx(arr, idx, value):
    """Set a single element, returning a new array."""
    out = arr.copy()
    out[idx] = value
    return out


# ---------------------------------------------------------------------------
# ArrayFeature subclasses (registered to "overcooked" scope)
# ---------------------------------------------------------------------------


@register_feature_type("overcooked_inventory", scope="overcooked")
class OvercookedInventory(ArrayFeature):
    per_agent = True
    obs_dim = 5

    @classmethod
    def build_feature_fn(cls, scope):
        from cogrid.backend import xp
        from cogrid.core.grid_object import object_to_idx

        inv_type_order = ["onion", "onion_soup", "plate", "tomato", "tomato_soup"]
        inv_type_ids = xp.array(
            [object_to_idx(name, scope=scope) for name in inv_type_order],
            dtype=xp.int32,
        )

        def fn(state_dict, agent_idx):
            return overcooked_inventory_feature(
                state_dict["agent_inv"], agent_idx, inv_type_ids,
            )
        return fn


@register_feature_type("next_to_counter", scope="overcooked")
class NextToCounter(ArrayFeature):
    per_agent = True
    obs_dim = 4

    @classmethod
    def build_feature_fn(cls, scope):
        from cogrid.core.grid_object import object_to_idx

        counter_type_id = object_to_idx("counter", scope=scope)

        def fn(state_dict, agent_idx):
            return next_to_counter_feature(
                state_dict["agent_pos"], agent_idx,
                state_dict["object_type_map"], counter_type_id,
            )
        return fn


@register_feature_type("next_to_pot", scope="overcooked")
class NextToPot(ArrayFeature):
    per_agent = True
    obs_dim = 16

    @classmethod
    def build_feature_fn(cls, scope):
        from cogrid.core.grid_object import object_to_idx

        pot_type_id = object_to_idx("pot", scope=scope)

        def fn(state_dict, agent_idx):
            return next_to_pot_feature(
                state_dict["agent_pos"], agent_idx,
                state_dict["object_type_map"], pot_type_id,
                state_dict["pot_positions"], state_dict["pot_contents"],
                state_dict["pot_timer"],
            )
        return fn


def _make_closest_obj_feature(obj_name, n_closest):
    """Factory to create and register a ClosestObj ArrayFeature variant."""
    feature_id = f"closest_{obj_name}"
    _obs_dim = 2 * n_closest
    _n = n_closest  # capture for closure
    _obj = obj_name  # capture for closure

    class _Cls(ArrayFeature):
        per_agent = True
        obs_dim = _obs_dim

        @classmethod
        def build_feature_fn(cls, scope):
            from cogrid.core.grid_object import object_to_idx

            target_type_id = object_to_idx(_obj, scope=scope)

            def fn(state_dict, agent_idx):
                return closest_obj_feature(
                    state_dict["agent_pos"], agent_idx,
                    state_dict["object_type_map"], state_dict["object_state_map"],
                    target_type_id, _n,
                )
            return fn

    _Cls.__name__ = f"Closest{obj_name.replace('_', ' ').title().replace(' ', '')}"
    _Cls.__qualname__ = _Cls.__name__

    # Register after setting class attributes (decorator validates obs_dim/per_agent)
    return register_feature_type(feature_id, scope="overcooked")(_Cls)


# Register all 7 ClosestObj variants (matching build_overcooked_feature_fn order)
_CLOSEST_OBJ_SPECS = [
    ("onion", 4), ("plate", 4), ("plate_stack", 2), ("onion_stack", 2),
    ("onion_soup", 4), ("delivery_zone", 2), ("counter", 4),
]
for _name, _n in _CLOSEST_OBJ_SPECS:
    _make_closest_obj_feature(_name, _n)


@register_feature_type("ordered_pot_features", scope="overcooked")
class OrderedPotFeatures(ArrayFeature):
    per_agent = True
    obs_dim = 24  # 12 features * max_num_pots=2

    @classmethod
    def build_feature_fn(cls, scope):
        from cogrid.core.grid_object import object_to_idx

        onion_id = object_to_idx("onion", scope=scope)
        tomato_id = object_to_idx("tomato", scope=scope)

        def fn(state_dict, agent_idx):
            return ordered_pot_features(
                state_dict["agent_pos"], agent_idx,
                state_dict["pot_positions"], state_dict["pot_contents"],
                state_dict["pot_timer"],
                max_num_pots=2, onion_id=onion_id, tomato_id=tomato_id,
            )
        return fn


@register_feature_type("dist_to_other_players", scope="overcooked")
class DistToOtherPlayers(ArrayFeature):
    per_agent = True
    obs_dim = 2  # 2 * (2 agents - 1) = 2

    @classmethod
    def build_feature_fn(cls, scope):
        def fn(state_dict, agent_idx):
            return dist_to_other_players_feature(
                state_dict["agent_pos"], agent_idx, n_agents=2,
            )
        return fn


@register_feature_type("layout_id", scope="overcooked")
class LayoutID(ArrayFeature):
    per_agent = False
    obs_dim = 5
    _layout_idx = 0  # Set before build_feature_fn; Phase 18 wires this

    @classmethod
    def build_feature_fn(cls, scope):
        idx = cls._layout_idx

        def fn(state_dict):
            return layout_id_feature(idx)
        return fn


@register_feature_type("environment_layout", scope="overcooked")
class EnvironmentLayout(ArrayFeature):
    per_agent = False
    obs_dim = 462  # 6 types * 11 * 7 (max layout shape)
    _max_layout_shape = (11, 7)

    @classmethod
    def build_feature_fn(cls, scope):
        from cogrid.core.grid_object import object_to_idx

        layout_type_names = ["counter", "pot", "onion", "plate", "onion_stack", "plate_stack"]
        layout_type_ids = [object_to_idx(name, scope=scope) for name in layout_type_names]
        max_shape = cls._max_layout_shape

        def fn(state_dict):
            return environment_layout_feature(
                state_dict["object_type_map"], layout_type_ids, max_shape,
            )
        return fn


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_overcooked_feature_fn(scope, n_agents, layout_idx, grid_shape, max_num_pots=2, max_layout_shape=(11, 7)):
    """Build the composed Overcooked feature function.

    Called at init time. Returns ``fn(state_dict, agent_idx) -> (obs_dim,) float32``.

    The observation is ego-centric: focal agent features first, then other
    agents in index order, then global features.

    Args:
        scope: Registry scope ("overcooked").
        n_agents: Number of agents.
        layout_idx: Integer index of the current layout (0-4).
        grid_shape: (H, W) of the current grid.
        max_num_pots: Maximum number of pots across layouts.
        max_layout_shape: (dim0, dim1) for environment layout feature, matching Python convention.
    """
    from cogrid.backend import xp
    from cogrid.core.grid_object import object_to_idx, build_lookup_tables

    # Pre-compute type IDs
    inv_type_order = ["onion", "onion_soup", "plate", "tomato", "tomato_soup"]
    inv_type_ids = xp.array(
        [object_to_idx(name, scope=scope) for name in inv_type_order],
        dtype=xp.int32,
    )

    counter_type_id = object_to_idx("counter", scope=scope)
    pot_type_id = object_to_idx("pot", scope=scope)
    onion_id = object_to_idx("onion", scope=scope)
    tomato_id = object_to_idx("tomato", scope=scope)

    # Type IDs for ClosestObj features (7 types, matching Python feature order)
    closest_obj_specs = [
        ("onion", 4),
        ("plate", 4),
        ("plate_stack", 2),
        ("onion_stack", 2),
        ("onion_soup", 4),
        ("delivery_zone", 2),
        ("counter", 4),
    ]
    closest_type_ids = [
        (object_to_idx(name, scope=scope), n_closest)
        for name, n_closest in closest_obj_specs
    ]

    # Type IDs for environment layout feature (6 types)
    layout_type_names = ["counter", "pot", "onion", "plate", "onion_stack", "plate_stack"]
    layout_type_ids_arr = [object_to_idx(name, scope=scope) for name in layout_type_names]

    tables = build_lookup_tables(scope=scope)
    can_overlap_table = xp.array(tables["CAN_OVERLAP"], dtype=xp.int32)

    # Pre-compute layout_id one-hot (constant for the episode)
    _layout_id = layout_id_feature(layout_idx)

    capacity = 3

    def _per_agent_features(state_dict, focal_idx):
        """Extract per-agent features for a single agent. (105,) for 2-agent case."""
        from cogrid.feature_space.array_features import (
            agent_dir_feature,
            agent_pos_feature,
            can_move_direction_feature,
        )

        parts = []

        # 1. AgentDir (4,)
        parts.append(agent_dir_feature(state_dict["agent_dir"], focal_idx))

        # 2. OvercookedInventory (5,)
        parts.append(overcooked_inventory_feature(
            state_dict["agent_inv"], focal_idx, inv_type_ids,
        ))

        # 3. NextToCounter (4,)
        parts.append(next_to_counter_feature(
            state_dict["agent_pos"], focal_idx,
            state_dict["object_type_map"], counter_type_id,
        ))

        # 4. NextToPot (16,)
        parts.append(next_to_pot_feature(
            state_dict["agent_pos"], focal_idx,
            state_dict["object_type_map"], pot_type_id,
            state_dict["pot_positions"], state_dict["pot_contents"],
            state_dict["pot_timer"], capacity=capacity,
        ))

        # 5-11. ClosestObj x7 types (44 total)
        for type_id, n_closest in closest_type_ids:
            parts.append(closest_obj_feature(
                state_dict["agent_pos"], focal_idx,
                state_dict["object_type_map"], state_dict["object_state_map"],
                type_id, n_closest,
            ))

        # 12. OrderedPotFeatures (24,)
        parts.append(ordered_pot_features(
            state_dict["agent_pos"], focal_idx,
            state_dict["pot_positions"], state_dict["pot_contents"],
            state_dict["pot_timer"], max_num_pots, onion_id, tomato_id,
            capacity=capacity,
        ))

        # 13. DistToOtherPlayers (2,)
        parts.append(dist_to_other_players_feature(
            state_dict["agent_pos"], focal_idx, n_agents,
        ))

        # 14. AgentPosition (2,)
        parts.append(agent_pos_feature(state_dict["agent_pos"], focal_idx))

        # 15. CanMoveDirection (4,)
        parts.append(can_move_direction_feature(
            state_dict["agent_pos"], focal_idx,
            state_dict["wall_map"], state_dict["object_type_map"],
            can_overlap_table,
        ))

        return xp.concatenate([p.ravel().astype(xp.float32) for p in parts])

    def _global_features(state_dict):
        """Extract global features. (467,)."""
        parts = []

        # 16. LayoutID (5,)
        parts.append(_layout_id)

        # 17. EnvironmentLayout (462,)
        parts.append(environment_layout_feature(
            state_dict["object_type_map"], layout_type_ids_arr, max_layout_shape,
        ))

        return xp.concatenate([p.ravel().astype(xp.float32) for p in parts])

    def feature_fn(state_dict, agent_idx):
        """Compose ego-centric observation: focal agent, others, globals."""
        parts = []

        # Focal agent first
        parts.append(_per_agent_features(state_dict, agent_idx))

        # Other agents in index order
        for i in range(n_agents):
            if i == agent_idx:
                continue
            parts.append(_per_agent_features(state_dict, i))

        # Global features
        parts.append(_global_features(state_dict))

        return xp.concatenate(parts)

    return feature_fn


# ---------------------------------------------------------------------------
# Register Overcooked feature order and layout indices
# (moved from cogrid/core/autowire.py -- domain-specific, not core logic)
# ---------------------------------------------------------------------------

from cogrid.core.component_registry import (
    register_feature_order,
    register_layout_indices,
)


def _overcooked_pre_compose_hook(layout_idx: int, scope: str) -> None:
    """Set LayoutID._layout_idx before feature composition."""
    LayoutID._layout_idx = layout_idx


register_feature_order("overcooked", [
    "agent_dir",
    "overcooked_inventory",
    "next_to_counter",
    "next_to_pot",
    "closest_onion",
    "closest_plate",
    "closest_plate_stack",
    "closest_onion_stack",
    "closest_onion_soup",
    "closest_delivery_zone",
    "closest_counter",
    "ordered_pot_features",
    "dist_to_other_players",
    "agent_position",
    "can_move_direction",
    "layout_id",
    "environment_layout",
], pre_compose_hook=_overcooked_pre_compose_hook)

register_layout_indices("overcooked", {
    "overcooked_cramped_room_v0": 0,
    "overcooked_asymmetric_advantages_v0": 1,
    "overcooked_coordination_ring_v0": 2,
    "overcooked_forced_coordination_v0": 3,
    "overcooked_counter_circuit_v0": 4,
})
