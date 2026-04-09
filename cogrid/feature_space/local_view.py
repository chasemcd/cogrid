"""Generic local-view feature for partial observability.

When ``local_view_radius`` is set, produces a (2r+1, 2r+1, C) multi-channel
spatial tensor centered on the focal agent.  When it is ``None``, produces
an (H, W, C) tensor of the full grid.

Object types and holdable items are auto-discovered from the component
registry.  Scope-specific extra channels (e.g. Overcooked pot state) are
added by subclassing and overriding ``extra_n_channels`` and
``build_extra_channel_fn``.

Channel layout (compact encoding):
    1. Object type ID (normalized)   (1 channel)
    2. Agent position                (A channels, focal first)
    3. Agent direction (normalized)  (A channels)
    4. Agent inventory (normalized)  (A channels)
    5. Object state map (normalized) (1 channel)
    6. Extra channels (subclass)     (E channels)

Total: 1 + A + A + A + 1 + E = 2 + 3A + E
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
    """Return sorted object type names for normalization range.

    Excludes None, "free_space", and agent direction pseudo-types.
    If ``env_config["local_view_type_names"]`` is set, use that instead.
    """
    if env_config is not None and "local_view_type_names" in env_config:
        return sorted(env_config["local_view_type_names"])

    from cogrid.core.objects.registry import get_object_names

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
# Channel count (compact encoding)
# ---------------------------------------------------------------------------

_N_TYPE_CHANNELS = 1  # single normalized type ID
_N_OSM_CHANNELS = 1  # single normalized object state
_N_DIR_CHANNELS_PER_AGENT = 1  # single normalized direction
_N_INV_CHANNELS_PER_AGENT = 1  # single normalized held type ID
_N_POS_CHANNELS_PER_AGENT = 1  # binary position indicator


def _n_core_channels(n_types, n_agents, n_holdable):
    per_agent = _N_POS_CHANNELS_PER_AGENT + _N_DIR_CHANNELS_PER_AGENT + _N_INV_CHANNELS_PER_AGENT
    return _N_TYPE_CHANNELS + per_agent * n_agents + _N_OSM_CHANNELS


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
    """Build a spatial observation tensor, flattened to 1-D.

    When *radius* is not None, produces a ``(2r+1, 2r+1, C)`` agent-centered
    window.  When *radius* is None, produces an ``(H, W, C)`` full-grid view.

    Uses compact encoding: single normalized channels instead of one-hot.

    *extra_channel_fn*, if not None, is called as
    ``extra_channel_fn(state, H, W) -> list[(H, W) float32]`` and the
    resulting layers are appended after the core channels.
    """
    H, W = state.object_type_map.shape

    if radius is not None:
        window = 2 * radius + 1
        pos = state.agent_pos[agent_idx]
        pad_width = ((radius, radius), (radius, radius))

        def _window(arr):
            return dynamic_slice_2d(xp.pad(arr, pad_width), pos[0], pos[1], window, window)
    else:

        def _window(arr):
            return arr

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

    # 1. Object type ID — single normalized channel
    n_types = max(type_ids) + 1 if type_ids else 1
    channels.append(_window(otm.astype(xp.float32) / n_types))

    # Agent ordering: focal first, then others ascending
    agent_order = [agent_idx] + [i for i in range(n_agents) if i != agent_idx]

    # 2. Agent position channels (1 per agent, binary)
    for ai in agent_order:
        ch = xp.zeros((H, W), dtype=xp.float32)
        a_pos = state.agent_pos[ai]
        valid = (a_pos[0] >= 0) & (a_pos[1] >= 0)
        ch = _set_cell(ch, a_pos[0], a_pos[1], xp.where(valid, 1.0, 0.0))
        channels.append(_window(ch))

    # 3. Agent direction — single normalized channel per agent
    for ai in agent_order:
        a_pos = state.agent_pos[ai]
        a_dir = state.agent_dir[ai]
        valid = (a_pos[0] >= 0) & (a_pos[1] >= 0)
        ch = xp.zeros((H, W), dtype=xp.float32)
        # 0.25=R, 0.5=D, 0.75=L, 1.0=U; 0=no agent
        val = xp.where(valid, (a_dir + 1).astype(xp.float32) / 4.0, 0.0)
        ch = _set_cell(ch, a_pos[0], a_pos[1], val)
        channels.append(_window(ch))

    # 4. Agent inventory — single normalized channel per agent
    for ai in agent_order:
        a_pos = state.agent_pos[ai]
        held = state.agent_inv[ai, 0]
        valid = (a_pos[0] >= 0) & (a_pos[1] >= 0)
        ch = xp.zeros((H, W), dtype=xp.float32)
        val = xp.where(
            valid & (held >= 0),
            (held + 1).astype(xp.float32) / n_types,
            0.0,
        )
        ch = _set_cell(ch, a_pos[0], a_pos[1], val)
        channels.append(_window(ch))

    # 5. Object state map — single normalized channel
    osm_max = (
        xp.float32(max(int(osm.max()), 1))
        if not hasattr(osm, "at")
        else xp.where(osm.max() > 0, osm.max().astype(xp.float32), xp.float32(1.0))
    )
    channels.append(_window(osm.astype(xp.float32) / osm_max))

    # 6. Extra channels from subclass
    if extra_channel_fn is not None:
        for layer in extra_channel_fn(state, H, W, agent_idx):
            channels.append(_window(layer))

    return xp.stack(channels, axis=-1).ravel()


# ---------------------------------------------------------------------------
# Feature class
# ---------------------------------------------------------------------------


@register_feature_type("local_view", scope="global")
class LocalView(Feature):
    """Generic local-view feature for any environment scope.

    Controlled by ``local_view_radius`` in env_config:

    * **numeric** — produces a ``(2r+1, 2r+1, C)`` agent-centered window,
      providing partial observability.
    * **None / absent** — produces an ``(H, W, C)`` full-grid tensor.
      Requires ``grid_height`` and ``grid_width`` in env_config so the
      observation dimension can be computed at build time.

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

        Returns None (no extras) or
        ``fn(state, H, W, agent_idx) -> list[(H,W) float32]``.
        """
        return None

    @classmethod
    def _get_radius(cls, env_config):
        """Resolve the local-view window radius.

        Returns the value of ``local_view_radius`` from *env_config*, or
        ``None`` when the key is absent or explicitly ``None`` (full grid).
        """
        if env_config is None:
            return None
        return env_config.get("local_view_radius")

    @classmethod
    def _get_spatial_dims(cls, env_config, radius):
        """Return ``(height, width)`` of the observation tensor."""
        if radius is not None:
            w = 2 * radius + 1
            return w, w
        # Full grid — read dimensions from config.
        h = env_config.get("grid_height")
        w = env_config.get("grid_width")
        if h is None or w is None:
            raise ValueError(
                "local_view with local_view_radius=None (full grid) requires "
                "'grid_height' and 'grid_width' in env_config."
            )
        return h, w

    @classmethod
    def compute_obs_dim(cls, scope, env_config=None):
        """Compute the flat observation dimension for a local view."""
        r = cls._get_radius(env_config)
        n_agents = env_config.get("n_agents", 2) if env_config else 2
        type_names = _discover_type_names(scope, env_config)
        holdable_names = _discover_holdable_names(scope, env_config)
        n_extra = cls.extra_n_channels(scope, env_config)

        h, w = cls._get_spatial_dims(env_config, r)
        n_ch = _n_core_channels(len(type_names), n_agents, len(holdable_names)) + n_extra
        return h * w * n_ch

    @classmethod
    def build_feature_fn(cls, scope, env_config=None):
        """Build a feature function that extracts a local view around an agent."""
        from cogrid.core.objects.registry import object_to_idx

        r = cls._get_radius(env_config)
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
