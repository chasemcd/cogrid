"""Overcooked feature extractors.

Produces the same 677-dim (for 2 agents) ego-centric observation as
the legacy OOP feature system, but operates entirely on state arrays
so it works with both numpy and JAX backends.

All functions are JIT-compatible: no Python control flow on traced
values, no int()/np.array() conversions of traced arrays.

Feature composition is handled by autowire via Feature subclasses
registered in this module.
"""

from cogrid.backend import xp
from cogrid.backend.array_ops import topk_smallest_indices
from cogrid.core.features import Feature, register_feature_type

# ---------------------------------------------------------------------------
# Order observation constants (defaults match _build_order_tables)
# ---------------------------------------------------------------------------

_ORDER_MAX_ACTIVE = 3
_ORDER_N_RECIPES = 2  # DEFAULT_RECIPES has 2 recipes
_ORDER_FEATURES_PER = _ORDER_N_RECIPES + 1  # recipe one-hot + normalized_time
_ORDER_OBS_DIM = _ORDER_MAX_ACTIVE * _ORDER_FEATURES_PER  # 3 * 3 = 9

# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def overcooked_inventory_feature(agent_inv, agent_idx, inv_type_ids):
    """One-hot inventory encoding. (5,) = [onion, onion_soup, plate, tomato, tomato_soup].

    Empty inventory (-1) matches nothing -> all zeros.
    """
    held = agent_inv[agent_idx, 0]
    return (inv_type_ids == held).astype(xp.int32)


def next_to_counter_feature(agent_pos, agent_idx, object_type_map, counter_type_id):
    """Multi-hot cardinal adjacency to counters. (4,) = [R, L, D, U]."""
    H, W = object_type_map.shape
    deltas = xp.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=xp.int32)
    pos = agent_pos[agent_idx]
    neighbors = pos[None, :] + deltas

    in_bounds = (
        (neighbors[:, 0] >= 0)
        & (neighbors[:, 0] < H)
        & (neighbors[:, 1] >= 0)
        & (neighbors[:, 1] < W)
    )

    clipped = xp.clip(neighbors, xp.array([0, 0]), xp.array([H - 1, W - 1]))
    types = object_type_map[clipped[:, 0], clipped[:, 1]]

    return (in_bounds & (types == counter_type_id)).astype(xp.int32)


def next_to_pot_feature(
    agent_pos,
    agent_idx,
    object_type_map,
    pot_type_id,
    pot_positions,
    pot_contents,
    pot_timer,
    capacity=3,
):
    """(16,) adjacency to pots, with status encoding.

    Layout: [empty(4), partial(4), cooking(4), ready(4)], indexed by direction.
    Fully vectorized over directions and pots — no Python control flow on traced values.
    """
    H, W = object_type_map.shape
    deltas = xp.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=xp.int32)
    pos = agent_pos[agent_idx]
    neighbors = pos[None, :] + deltas  # (4, 2)

    in_bounds = (
        (neighbors[:, 0] >= 0)
        & (neighbors[:, 0] < H)
        & (neighbors[:, 1] >= 0)
        & (neighbors[:, 1] < W)
    )

    clipped = xp.clip(neighbors, xp.array([0, 0]), xp.array([H - 1, W - 1]))
    types = object_type_map[clipped[:, 0], clipped[:, 1]]
    is_pot = in_bounds & (types == pot_type_id)  # (4,)

    # For each direction, find which pot it matches (if any)
    # neighbor_rc: (4, 1, 2) vs pot_positions: (1, n_pots, 2)
    neighbor_rc = clipped[:, None, :]  # (4, 1, 2)
    pot_rc = pot_positions[None, :, :]  # (1, n_pots, 2)
    match = (neighbor_rc[:, :, 0] == pot_rc[:, :, 0]) & (
        neighbor_rc[:, :, 1] == pot_rc[:, :, 1]
    )  # (4, n_pots)

    # Count items per pot and get timer: (n_pots,)
    items_per_pot = xp.sum(pot_contents > 0, axis=1)  # (n_pots,)

    # Pot status per pot: [empty, partial, cooking, ready]
    is_empty = items_per_pot == 0  # (n_pots,)
    is_partial = (items_per_pot > 0) & (items_per_pot < capacity)  # (n_pots,)
    is_cooking = (items_per_pot == capacity) & (pot_timer > 0)  # (n_pots,)
    is_ready = (items_per_pot == capacity) & (pot_timer == 0)  # (n_pots,)

    # For each direction, check if any matched pot has each status
    # match: (4, n_pots), is_empty: (n_pots,) -> broadcast to (4, n_pots)
    dir_empty = xp.any(match & is_empty[None, :], axis=1)  # (4,)
    dir_partial = xp.any(match & is_partial[None, :], axis=1)  # (4,)
    dir_cooking = xp.any(match & is_cooking[None, :], axis=1)  # (4,)
    dir_ready = xp.any(match & is_ready[None, :], axis=1)  # (4,)

    # Mask by is_pot
    result = xp.concatenate(
        [
            (is_pot & dir_empty).astype(xp.int32),
            (is_pot & dir_partial).astype(xp.int32),
            (is_pot & dir_cooking).astype(xp.int32),
            (is_pot & dir_ready).astype(xp.int32),
        ]
    )

    return result


def closest_obj_feature(
    agent_pos, agent_idx, object_type_map, object_state_map, target_type_id, n, large_dist=9999
):
    """(2*n,) deltas (dy, dx) to the n closest instances of target_type_id.

    Searches both object_type_map (for placed objects like pots/stacks) and
    object_state_map (for items placed on counters).
    Fully vectorized — uses argsort with masked distances.
    """
    H, W = object_type_map.shape
    pos = agent_pos[agent_idx]  # (2,)

    type_flat = object_type_map.ravel()  # (H*W,)
    state_flat = object_state_map.ravel()  # (H*W,)
    match = (type_flat == target_type_id) | (state_flat == target_type_id)

    # Coordinate arrays
    rows = xp.arange(H, dtype=xp.int32).repeat(W)
    cols = xp.tile(xp.arange(W, dtype=xp.int32), H)

    # Exclude agent's own position
    agent_mask = ~((rows == pos[0]) & (cols == pos[1]))
    match = match & agent_mask

    # Deltas and manhattan distances
    dys = pos[0] - rows  # (H*W,)
    dxs = pos[1] - cols  # (H*W,)
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
    return xp.stack([top_dys, top_dxs], axis=1).ravel().astype(xp.int32)


# Specs for merged closest_objects feature, sorted alphabetically to match
# the compose order of the individual closest_* features.
_CLOSEST_SPECS_SORTED = [
    ("counter", 4),
    ("delivery_zone", 2),
    ("onion", 4),
    ("onion_soup", 4),
    ("onion_stack", 2),
    ("plate", 4),
    ("plate_stack", 2),
]


def closest_objects_feature(
    agent_pos, agent_idx, object_type_map, object_state_map, type_ids_and_ns, large_dist=9999
):
    """(44,) deltas to closest instances of 7 object types in one pass.

    Shares coordinate/distance computation across all types and uses
    top-k selection (single HLO op on JAX) instead of sequential argmin.

    type_ids_and_ns: list of (type_id, n_closest) tuples, in alphabetical order.
    """
    H, W = object_type_map.shape
    pos = agent_pos[agent_idx]  # (2,)

    type_flat = object_type_map.ravel()  # (H*W,)
    state_flat = object_state_map.ravel()  # (H*W,)

    # Shared coordinate arrays (computed once)
    rows = xp.arange(H, dtype=xp.int32).repeat(W)
    cols = xp.tile(xp.arange(W, dtype=xp.int32), H)

    # Exclude agent's own position
    agent_mask = ~((rows == pos[0]) & (cols == pos[1]))

    # Shared deltas and distances
    dys = pos[0] - rows  # (H*W,)
    dxs = pos[1] - cols  # (H*W,)
    dists = xp.abs(dys) + xp.abs(dxs)  # (H*W,)

    parts = []
    for type_id, n in type_ids_and_ns:
        match = ((type_flat == type_id) | (state_flat == type_id)) & agent_mask

        # Mask non-matches with large distance, then find k smallest in one op
        masked_dists = xp.where(match, dists, large_dist)
        top_indices = topk_smallest_indices(masked_dists, n)

        top_dys = dys[top_indices]
        top_dxs = dxs[top_indices]
        top_valid = masked_dists[top_indices] < large_dist

        top_dys = xp.where(top_valid, top_dys, 0)
        top_dxs = xp.where(top_valid, top_dxs, 0)

        parts.append(xp.stack([top_dys, top_dxs], axis=1).ravel())

    return xp.concatenate(parts).astype(xp.int32)


def ordered_pot_features(
    agent_pos,
    agent_idx,
    pot_positions,
    pot_contents,
    pot_timer,
    max_num_pots,
    onion_id,
    tomato_id,
    capacity=3,
):
    """(12 * max_num_pots,) per-pot features in grid-scan order.

    Per pot: [reachable(1), status(4), contents(2), timer(1), distance(2), location(2)] = 12.
    Status one-hot: [ready, empty, full/cooking, partial] matching _calc_pot_features order.
    Fully vectorized — no Python control flow on traced values.
    """
    pos = agent_pos[agent_idx]  # (2,)

    # Items per pot: (n_pots,)
    n_items = xp.sum(pot_contents > 0, axis=1)

    # Status one-hot per pot: [ready, empty, full/cooking, partial]
    is_ready = ((n_items == capacity) & (pot_timer == 0)).astype(xp.float32)
    is_empty = (n_items == 0).astype(xp.float32)
    is_cooking = ((n_items == capacity) & (pot_timer > 0)).astype(xp.float32)
    is_partial = ((n_items > 0) & (n_items < capacity)).astype(xp.float32)

    # Contents per pot: count of onion and tomato
    n_onion = xp.sum(pot_contents == onion_id, axis=1).astype(xp.float32)  # (n_pots,)
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
    per_pot = xp.stack(
        [
            reachable,
            is_ready,
            is_empty,
            is_cooking,
            is_partial,
            n_onion,
            n_tomato,
            timer_val,
            dy,
            dx,
            pr,
            pc,
        ],
        axis=1,
    )  # (n_pots, 12)

    # Pad to max_num_pots (if fewer pots than max)
    n_pots = pot_positions.shape[0]
    if n_pots < max_num_pots:
        pad = xp.zeros((max_num_pots - n_pots, 12), dtype=xp.float32)
        per_pot = xp.concatenate([per_pot, pad], axis=0)

    # Take only max_num_pots and flatten
    return per_pot[:max_num_pots].ravel()


def dist_to_other_players_feature(agent_pos, agent_idx, n_agents):
    """(2 * (n_agents - 1),) distance to other agents."""
    pos = agent_pos[agent_idx]
    # Build mask of other agents (agent_idx is a Python int, safe under JIT)
    others = [i for i in range(n_agents) if i != agent_idx]
    other_pos = xp.stack([agent_pos[i] for i in others])  # (n_agents-1, 2)
    dys = pos[0] - other_pos[:, 0]
    dxs = pos[1] - other_pos[:, 1]
    return xp.stack([dys, dxs], axis=1).ravel().astype(xp.int32)


def layout_id_feature(layout_idx, num_layouts=5):
    """(num_layouts,) one-hot layout identifier."""
    return (xp.arange(num_layouts) == layout_idx).astype(xp.int32)


def environment_layout_feature(object_type_map, layout_type_ids, max_shape):
    """(6 * max_shape[0] * max_shape[1],) binary masks for each of 6 object types.

    Types: [counter, pot, onion, plate, onion_stack, plate_stack].
    Uses the same (max_shape[0], max_shape[1]) convention and flat indexing
    as the Python ``EnvironmentLayout`` feature: ``flat_index = row * max_shape[1] + col``.
    Vectorized using scatter-style indexing (no Python loops on traced values).
    """
    H, W = object_type_map.shape
    dim0, dim1 = max_shape
    channel_size = dim0 * dim1
    n_types = len(layout_type_ids)

    # Build flat coordinate indices: flat_index[r, c] = r * dim1 + c
    row_idx = xp.arange(H, dtype=xp.int32)[:, None] * dim1  # (H, 1)
    col_idx = xp.arange(W, dtype=xp.int32)[None, :]  # (1, W)
    flat_idx = (row_idx + col_idx).ravel()  # (H*W,)

    otm_flat = object_type_map.ravel()  # (H*W,)

    result = xp.zeros(n_types * channel_size, dtype=xp.int32)

    for t_idx in range(n_types):
        type_id = layout_type_ids[t_idx]
        offset = t_idx * channel_size
        is_match = (otm_flat == type_id).astype(xp.int32)  # (H*W,)
        # Scatter matched positions into the result
        target_indices = flat_idx + offset  # (H*W,)
        if hasattr(result, "at"):  # JAX
            result = result.at[target_indices].add(is_match)
        else:
            result[target_indices[is_match.astype(bool)]] = 1

    return result


# ---------------------------------------------------------------------------
# Feature subclasses (registered to "overcooked" scope)
# ---------------------------------------------------------------------------


@register_feature_type("order_observation", scope="overcooked")
class OrderObservation(Feature):
    """Encodes active orders as recipe one-hot + normalized time remaining.

    Output: (_ORDER_OBS_DIM,) = max_active * (n_recipes + 1) = 9 by default.
    Returns zeros when orders are not configured (backward compat).

    When *env_config* is provided, ``"recipes"`` length and
    ``"orders"["max_active"]`` override the defaults.
    """

    per_agent = False  # Orders are global state, same for all agents
    obs_dim = _ORDER_OBS_DIM  # 9

    @classmethod
    def compute_obs_dim(cls, scope, env_config=None):
        """Return obs_dim from env_config recipe/order counts, or default."""
        if env_config is not None:
            n_recipes = len(env_config["recipes"]) if "recipes" in env_config else _ORDER_N_RECIPES
            max_active = (
                env_config["orders"].get("max_active", _ORDER_MAX_ACTIVE)
                if "orders" in env_config
                else _ORDER_MAX_ACTIVE
            )
            return max_active * (n_recipes + 1)
        return cls.obs_dim

    @classmethod
    def build_feature_fn(cls, scope, env_config=None):
        """Build order observation feature function."""
        if env_config is not None and "recipes" in env_config:
            n_recipes = len(env_config["recipes"])
        else:
            n_recipes = _ORDER_N_RECIPES

        if env_config is not None and "orders" in env_config:
            max_active = env_config["orders"].get("max_active", _ORDER_MAX_ACTIVE)
            time_limit = xp.float32(env_config["orders"].get("time_limit", 200.0))
        else:
            max_active = _ORDER_MAX_ACTIVE
            time_limit = xp.float32(200.0)

        total_dim = max_active * (n_recipes + 1)

        def fn(state):
            order_recipe = getattr(state, "order_recipe", None)
            if order_recipe is None:
                return xp.zeros(total_dim, dtype=xp.float32)

            order_timer = state.order_timer
            active = order_recipe >= 0  # (max_active,) bool

            parts = []
            for i in range(max_active):
                # Recipe type one-hot (n_recipes,)
                recipe_onehot = xp.where(
                    active[i],
                    (xp.arange(n_recipes) == order_recipe[i]).astype(xp.float32),
                    xp.zeros(n_recipes, dtype=xp.float32),
                )
                # Normalized time remaining (1,)
                norm_time = xp.where(
                    active[i],
                    order_timer[i].astype(xp.float32) / time_limit,
                    xp.float32(0.0),
                )
                parts.append(recipe_onehot)
                parts.append(xp.array([norm_time], dtype=xp.float32))

            return xp.concatenate(parts)

        return fn


_DEFAULT_PICKUPABLE_TYPES = ["onion", "onion_soup", "plate", "tomato", "tomato_soup"]


@register_feature_type("overcooked_inventory", scope="overcooked")
class OvercookedInventory(Feature):
    """One-hot inventory encoding feature.

    Reads pickupable types from ``env_config["pickupable_types"]`` when
    available, otherwise uses ``_DEFAULT_PICKUPABLE_TYPES``.  This
    ensures the observation dimension is deterministic per config and
    independent of the global component registry.
    """

    per_agent = True
    obs_dim = len(_DEFAULT_PICKUPABLE_TYPES)  # 5

    @classmethod
    def compute_obs_dim(cls, scope, env_config=None):
        """Return obs_dim from env_config pickupable_types, or default."""
        if env_config is not None and "pickupable_types" in env_config:
            return len(env_config["pickupable_types"])
        return cls.obs_dim

    @classmethod
    def build_feature_fn(cls, scope, env_config=None):
        """Build the inventory feature function for the given scope."""
        from cogrid.core.grid_object import object_to_idx

        if env_config is not None and "pickupable_types" in env_config:
            pickupable_names = sorted(env_config["pickupable_types"])
        else:
            pickupable_names = sorted(_DEFAULT_PICKUPABLE_TYPES)

        inv_type_ids = xp.array(
            [object_to_idx(name, scope=scope) for name in pickupable_names],
            dtype=xp.int32,
        )

        def fn(state, agent_idx):
            return overcooked_inventory_feature(
                state.agent_inv,
                agent_idx,
                inv_type_ids,
            )

        return fn


@register_feature_type("next_to_counter", scope="overcooked")
class NextToCounter(Feature):
    """Cardinal adjacency to counters feature."""

    per_agent = True
    obs_dim = 4

    @classmethod
    def build_feature_fn(cls, scope):
        """Build the counter adjacency feature function."""
        from cogrid.core.grid_object import object_to_idx

        counter_type_id = object_to_idx("counter", scope=scope)

        def fn(state, agent_idx):
            return next_to_counter_feature(
                state.agent_pos,
                agent_idx,
                state.object_type_map,
                counter_type_id,
            )

        return fn


@register_feature_type("next_to_pot", scope="overcooked")
class NextToPot(Feature):
    """Cardinal adjacency to pots with status encoding."""

    per_agent = True
    obs_dim = 16

    @classmethod
    def build_feature_fn(cls, scope):
        """Build the pot adjacency feature function."""
        from cogrid.core.grid_object import object_to_idx

        pot_type_id = object_to_idx("pot", scope=scope)

        def fn(state, agent_idx):
            return next_to_pot_feature(
                state.agent_pos,
                agent_idx,
                state.object_type_map,
                pot_type_id,
                state.pot_positions,
                state.pot_contents,
                state.pot_timer,
            )

        return fn


def _make_closest_obj_feature(obj_name, n_closest):
    """Factory to create and register a ClosestObj Feature variant."""
    feature_id = f"closest_{obj_name}"
    _obs_dim = 2 * n_closest
    _n = n_closest  # capture for closure
    _obj = obj_name  # capture for closure

    class _Cls(Feature):
        per_agent = True
        obs_dim = _obs_dim

        @classmethod
        def build_feature_fn(cls, scope):
            from cogrid.core.grid_object import object_to_idx

            target_type_id = object_to_idx(_obj, scope=scope)

            def fn(state, agent_idx):
                return closest_obj_feature(
                    state.agent_pos,
                    agent_idx,
                    state.object_type_map,
                    state.object_state_map,
                    target_type_id,
                    _n,
                )

            return fn

    _Cls.__name__ = f"Closest{obj_name.replace('_', ' ').title().replace(' ', '')}"
    _Cls.__qualname__ = _Cls.__name__

    # Register after setting class attributes (decorator validates obs_dim/per_agent)
    return register_feature_type(feature_id, scope="overcooked")(_Cls)


# Register all 7 ClosestObj variants (matching build_overcooked_feature_fn order)
_CLOSEST_OBJ_SPECS = [
    ("onion", 4),
    ("plate", 4),
    ("plate_stack", 2),
    ("onion_stack", 2),
    ("onion_soup", 4),
    ("delivery_zone", 2),
    ("counter", 4),
]
for _name, _n in _CLOSEST_OBJ_SPECS:
    _make_closest_obj_feature(_name, _n)


@register_feature_type("closest_objects", scope="overcooked")
class ClosestObjects(Feature):
    """Merged closest-object feature for all 7 types in one pass."""

    per_agent = True
    obs_dim = 44  # 2*(4+2+4+4+2+4+2)

    @classmethod
    def build_feature_fn(cls, scope):
        """Build closest-object feature function."""
        from cogrid.core.grid_object import object_to_idx

        type_ids_and_ns = [
            (object_to_idx(name, scope=scope), n) for name, n in _CLOSEST_SPECS_SORTED
        ]

        def fn(state, agent_idx):
            return closest_objects_feature(
                state.agent_pos,
                agent_idx,
                state.object_type_map,
                state.object_state_map,
                type_ids_and_ns,
            )

        return fn


# ---------------------------------------------------------------------------
# Object type masks — binary spatial encoding (replaces closest_* for perf)
# ---------------------------------------------------------------------------

# Types to encode, alphabetical order.
_TYPE_MASK_NAMES = [
    "counter",
    "delivery_zone",
    "onion",
    "onion_soup",
    "onion_stack",
    "plate",
    "plate_stack",
]
_TYPE_MASK_MAX_H = 7  # max rows across all Overcooked layouts
_TYPE_MASK_MAX_W = 11  # max cols across all Overcooked layouts
_TYPE_MASK_CELLS = _TYPE_MASK_MAX_H * _TYPE_MASK_MAX_W  # 77


def object_type_masks_feature(object_type_map, object_state_map, type_ids):
    """(7 * 77,) binary masks for 7 object types, zero-padded to max layout size.

    Each channel is a max_H × max_W binary grid indicating where a type is
    present (in object_type_map or object_state_map).  Pure elementwise ops —
    no distances, no sorting, no selection.
    """
    H, W = object_type_map.shape
    pad_h = _TYPE_MASK_MAX_H - H
    pad_w = _TYPE_MASK_MAX_W - W

    parts = []
    for type_id in type_ids:
        mask = ((object_type_map == type_id) | (object_state_map == type_id)).astype(xp.int32)
        if pad_h > 0 or pad_w > 0:
            mask = xp.pad(mask, ((0, pad_h), (0, pad_w)))
        parts.append(mask.ravel())
    return xp.concatenate(parts)


@register_feature_type("object_type_masks", scope="overcooked")
class ObjectTypeMasks(Feature):
    """Binary spatial masks for 7 object types."""

    per_agent = False
    obs_dim = len(_TYPE_MASK_NAMES) * _TYPE_MASK_CELLS  # 7 * 77 = 539

    @classmethod
    def build_feature_fn(cls, scope):
        """Build object-type mask feature function."""
        from cogrid.core.grid_object import object_to_idx

        type_ids = [object_to_idx(name, scope=scope) for name in _TYPE_MASK_NAMES]

        def fn(state):
            return object_type_masks_feature(
                state.object_type_map,
                state.object_state_map,
                type_ids,
            )

        return fn


@register_feature_type("ordered_pot_features", scope="overcooked")
class OrderedPotFeatures(Feature):
    """Per-pot features in grid-scan order."""

    per_agent = True
    obs_dim = 24  # 12 features * max_num_pots=2

    @classmethod
    def build_feature_fn(cls, scope):
        """Build the ordered pot feature function."""
        from cogrid.core.grid_object import object_to_idx

        onion_id = object_to_idx("onion", scope=scope)
        tomato_id = object_to_idx("tomato", scope=scope)

        def fn(state, agent_idx):
            return ordered_pot_features(
                state.agent_pos,
                agent_idx,
                state.pot_positions,
                state.pot_contents,
                state.pot_timer,
                max_num_pots=2,
                onion_id=onion_id,
                tomato_id=tomato_id,
            )

        return fn


@register_feature_type("dist_to_other_players", scope="overcooked")
class DistToOtherPlayers(Feature):
    """Distance to other agents feature."""

    per_agent = True
    obs_dim = 2  # 2 * (2 agents - 1) = 2

    @classmethod
    def build_feature_fn(cls, scope):
        """Build the distance-to-others feature function."""

        def fn(state, agent_idx):
            return dist_to_other_players_feature(
                state.agent_pos,
                agent_idx,
                n_agents=2,
            )

        return fn


@register_feature_type("layout_id", scope="overcooked")
class LayoutID(Feature):
    """One-hot layout identifier feature."""

    per_agent = False
    obs_dim = 5
    _layout_idx = 0  # Set before build_feature_fn; Phase 18 wires this

    @classmethod
    def build_feature_fn(cls, scope):
        """Build the layout ID feature function."""
        idx = cls._layout_idx

        def fn(state):
            return layout_id_feature(idx)

        return fn


@register_feature_type("environment_layout", scope="overcooked")
class EnvironmentLayout(Feature):
    """Binary masks for object types across the layout."""

    per_agent = False
    obs_dim = 462  # 6 types * 11 * 7 (max layout shape)
    _max_layout_shape = (11, 7)

    @classmethod
    def build_feature_fn(cls, scope):
        """Build the environment layout feature function."""
        from cogrid.core.grid_object import object_to_idx

        layout_type_names = ["counter", "pot", "onion", "plate", "onion_stack", "plate_stack"]
        layout_type_ids = [object_to_idx(name, scope=scope) for name in layout_type_names]
        max_shape = cls._max_layout_shape

        def fn(state):
            return environment_layout_feature(
                state.object_type_map,
                layout_type_ids,
                max_shape,
            )

        return fn


# ---------------------------------------------------------------------------
# Register Overcooked feature order and layout indices
# (moved from cogrid/core/autowire.py -- domain-specific, not core logic)
# ---------------------------------------------------------------------------

from cogrid.core.component_registry import (  # noqa: E402
    register_layout_indices,
    register_pre_compose_hook,
)


def _overcooked_pre_compose_hook(layout_idx: int, scope: str, env_config=None) -> None:
    """Set LayoutID._layout_idx before feature composition."""
    LayoutID._layout_idx = layout_idx


register_pre_compose_hook("overcooked", _overcooked_pre_compose_hook)

register_layout_indices(
    "overcooked",
    {
        "overcooked_cramped_room_v0": 0,
        "overcooked_asymmetric_advantages_v0": 1,
        "overcooked_coordination_ring_v0": 2,
        "overcooked_forced_coordination_v0": 3,
        "overcooked_counter_circuit_v0": 4,
    },
)
