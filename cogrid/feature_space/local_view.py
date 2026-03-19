"""Generic local-view feature for partial observability.

Produces a (2r+1, 2r+1, C) multi-channel spatial tensor centered on the
focal agent, auto-discovering object types and holdable items from the
component registry.  Scope-specific extra channels (e.g. Overcooked pot
state) are added by subclassing and overriding ``extra_n_channels`` and
``build_extra_channel_fn``.

Channel layout:
    1. Object type binary masks  (T channels)
    2. Agent position             (A channels, focal first)
    3. Agent direction one-hot    (4A channels)
    4. Agent inventory one-hot    (H*A channels)
    5. Object state map raw       (1 channel)
    6. Extra channels (subclass)  (E channels)

Total: T + A + 4A + H*A + 1 + E
"""

from cogrid.backend import xp
from cogrid.backend.array_ops import dynamic_slice_2d
from cogrid.core.component_registry import (
    get_all_components,
    register_feature_type,
)
from cogrid.core.features import Feature

# ---------------------------------------------------------------------------
# Discovery helpers (build-time, not traced)
# ---------------------------------------------------------------------------


def _discover_type_names(scope, env_config=None):
    """Return sorted object type names for binary mask channels.

    Excludes None, "free_space", and agent direction pseudo-types.
    If ``env_config["local_view_type_names"]`` is set, use that instead.
    """
    if env_config is not None and "local_view_type_names" in env_config:
        return sorted(env_config["local_view_type_names"])

    from cogrid.core.grid_object import get_object_names

    all_names = get_object_names(scope)
    excluded = {None, "free_space"}
    return [
        n
        for n in all_names
        if n not in excluded and not (isinstance(n, str) and n.startswith("agent_"))
    ]


def _discover_holdable_names(scope, env_config=None):
    """Return sorted holdable type names for inventory channels.

    If ``env_config["pickupable_types"]`` is set, use that instead.
    Scans global + scope components for ``can_pickup == True``.
    """
    if env_config is not None and "pickupable_types" in env_config:
        return sorted(env_config["pickupable_types"])

    holdable = set()
    for s in ("global", scope) if scope != "global" else ("global",):
        for meta in get_all_components(s):
            if meta.properties.get("can_pickup", False):
                holdable.add(meta.object_id)
    return sorted(holdable)


# ---------------------------------------------------------------------------
# Channel count
# ---------------------------------------------------------------------------


def _n_core_channels(n_types, n_agents, n_holdable):
    return n_types + n_agents + 4 * n_agents + n_holdable * n_agents + 1


# ---------------------------------------------------------------------------
# Core feature function
# ---------------------------------------------------------------------------


def generic_local_view_feature(
    state,
    agent_idx,
    n_agents,
    radius,
    type_ids,
    holdable_ids,
    extra_channel_fn,
):
    """Build a (2r+1, 2r+1, C) local-view tensor, flattened to 1-D.

    *extra_channel_fn*, if not None, is called as
    ``extra_channel_fn(state, H, W) -> list[(H, W) float32]`` and the
    resulting layers are appended after the core channels.
    """
    H, W = state.object_type_map.shape
    window = 2 * radius + 1
    pos = state.agent_pos[agent_idx]
    pad_width = ((radius, radius), (radius, radius))

    def _window(arr):
        return dynamic_slice_2d(xp.pad(arr, pad_width), pos[0], pos[1], window, window)

    def _set_cell(ch, row, col, val):
        if hasattr(ch, "at"):  # JAX
            return ch.at[xp.clip(row, 0, H - 1), xp.clip(col, 0, W - 1)].set(val)
        else:
            r_i, c_i = int(row), int(col)
            if 0 <= r_i < H and 0 <= c_i < W:
                ch[r_i, c_i] = float(val)
            return ch

    channels = []
    otm = state.object_type_map
    osm = state.object_state_map

    # 1. Object type binary masks
    for tid in type_ids:
        channels.append(_window(((otm == tid) | (osm == tid)).astype(xp.float32)))

    # Agent ordering: focal first, then others ascending
    agent_order = [agent_idx] + [i for i in range(n_agents) if i != agent_idx]

    # 2. Agent position channels
    for ai in agent_order:
        ch = xp.zeros((H, W), dtype=xp.float32)
        a_pos = state.agent_pos[ai]
        valid = (a_pos[0] >= 0) & (a_pos[1] >= 0)
        ch = _set_cell(ch, a_pos[0], a_pos[1], xp.where(valid, 1.0, 0.0))
        channels.append(_window(ch))

    # 3. Agent direction channels (4 per agent)
    for ai in agent_order:
        a_pos = state.agent_pos[ai]
        a_dir = state.agent_dir[ai]
        valid = (a_pos[0] >= 0) & (a_pos[1] >= 0)
        for d in range(4):
            ch = xp.zeros((H, W), dtype=xp.float32)
            val = xp.where(valid & (a_dir == d), 1.0, 0.0)
            ch = _set_cell(ch, a_pos[0], a_pos[1], val)
            channels.append(_window(ch))

    # 4. Agent inventory channels (n_holdable per agent)
    for ai in agent_order:
        a_pos = state.agent_pos[ai]
        held = state.agent_inv[ai, 0]
        valid = (a_pos[0] >= 0) & (a_pos[1] >= 0)
        for hid in holdable_ids:
            ch = xp.zeros((H, W), dtype=xp.float32)
            val = xp.where(valid & (held == hid), 1.0, 0.0)
            ch = _set_cell(ch, a_pos[0], a_pos[1], val)
            channels.append(_window(ch))

    # 5. Object state map (1 channel)
    channels.append(_window(osm.astype(xp.float32)))

    # 6. Extra channels from subclass
    if extra_channel_fn is not None:
        for layer in extra_channel_fn(state, H, W):
            channels.append(_window(layer))

    return xp.stack(channels, axis=-1).ravel()


# ---------------------------------------------------------------------------
# Feature class
# ---------------------------------------------------------------------------


@register_feature_type("local_view", scope="global")
class LocalView(Feature):
    """Generic local-view feature for any environment scope.

    Auto-discovers object types and holdable items from the registry.
    Subclasses can override ``extra_n_channels`` and ``build_extra_channel_fn``
    to add scope-specific channels (e.g. pot state for Overcooked).
    """

    per_agent = True
    focal_only = True
    obs_dim = 0  # computed dynamically

    @classmethod
    def extra_n_channels(cls, scope, env_config=None):
        """Number of extra channels added by this class. Override in subclasses."""
        return 0

    @classmethod
    def build_extra_channel_fn(cls, scope, env_config=None):
        """Build extra-channel function. Override in subclasses.

        Returns None (no extras) or ``fn(state, H, W) -> list[(H,W) float32]``.
        """
        return None

    @classmethod
    def compute_obs_dim(cls, scope, env_config=None):
        r = env_config.get("observable_radius", 3) if env_config else 3
        n_agents = env_config.get("n_agents", 2) if env_config else 2
        type_names = _discover_type_names(scope, env_config)
        holdable_names = _discover_holdable_names(scope, env_config)
        n_extra = cls.extra_n_channels(scope, env_config)

        window = 2 * r + 1
        n_ch = _n_core_channels(len(type_names), n_agents, len(holdable_names)) + n_extra
        return window * window * n_ch

    @classmethod
    def build_feature_fn(cls, scope, env_config=None):
        from cogrid.core.grid_object import object_to_idx

        r = env_config.get("observable_radius", 3) if env_config else 3
        n_agents = env_config.get("n_agents", 2) if env_config else 2

        type_names = _discover_type_names(scope, env_config)
        type_ids = [object_to_idx(name, scope=scope) for name in type_names]

        holdable_names = _discover_holdable_names(scope, env_config)
        holdable_ids = [object_to_idx(name, scope=scope) for name in holdable_names]

        extra_channel_fn = cls.build_extra_channel_fn(scope, env_config)

        def fn(state, agent_idx):
            return generic_local_view_feature(
                state,
                agent_idx,
                n_agents,
                r,
                type_ids,
                holdable_ids,
                extra_channel_fn,
            )

        return fn
