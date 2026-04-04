"""Feature base class for feature components.

Each feature is a subclass with class attributes ``per_agent`` and ``obs_dim``,
and a ``build_feature_fn`` classmethod that returns a pure function.

Per-agent features: ``fn(state, agent_idx) -> ndarray``
Global features: ``fn(state) -> ndarray``

Feature composition is handled by ``compose_feature_fns()`` in this module.

Environment-specific features live in their respective envs/ modules.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cogrid.core.typing import ArrayLike

# Re-export for convenience (decorator lives in component_registry)
from cogrid.backend import xp
from cogrid.core.component_registry import register_feature_type  # noqa: F401


class Feature:
    """Base class for feature extractors.

    Subclasses MUST define:
        - ``per_agent``: bool class attribute. True if feature is per-agent,
          False if global.
        - ``obs_dim``: int class attribute. Dimensionality of the feature
          output after ravel().
        - ``build_feature_fn(cls, scope)``: classmethod that returns a pure
          function. For per_agent=True: fn(state, agent_idx) -> ndarray.
          For per_agent=False: fn(state) -> ndarray.

    ``state`` is a :class:`~cogrid.backend.state_view.StateView` —
    a frozen dataclass with dot access for core fields (``agent_pos``,
    ``agent_dir``, etc.) and ``__getattr__`` fallthrough for extras.

    Usage::

        @register_feature_type("agent_dir", scope="global")
        class AgentDir(Feature):
            per_agent = True
            obs_dim = 4

            @classmethod
            def build_feature_fn(cls, scope):
                def fn(state, agent_idx):
                    from cogrid.backend import xp

                    return (xp.arange(4) == state.agent_dir[agent_idx]).astype(xp.int32)

                return fn
    """

    per_agent: bool
    obs_dim: int

    @classmethod
    def compute_obs_dim(cls, scope: str, env_config: dict[str, Any] | None = None) -> int:
        """Compute observation dimension, optionally using env_config.

        Subclasses may override to return a config-dependent dimension.
        Default returns the static ``cls.obs_dim``.
        """
        return cls.obs_dim

    @classmethod
    def build_feature_fn(cls, scope: str) -> Callable[..., ArrayLike]:
        """Build and return the feature extraction function.

        Must return ``fn(state, agent_idx) -> ndarray`` for per-agent
        features, or ``fn(state) -> ndarray`` for global features.

        Subclasses that need config can use the signature
        ``build_feature_fn(cls, scope, env_config=None)``.
        """
        raise NotImplementedError(
            f"{cls.__name__}.build_feature_fn() is not implemented. "
            f"Subclasses must override build_feature_fn()."
        )


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def _call_build_feature_fn(
    cls: type[Feature], scope: str, env_config: dict[str, Any] | None
) -> Callable[..., ArrayLike]:
    """Call cls.build_feature_fn, passing env_config if the method accepts it."""
    sig = inspect.signature(cls.build_feature_fn)
    if "env_config" in sig.parameters:
        return cls.build_feature_fn(scope, env_config=env_config)
    return cls.build_feature_fn(scope)


def _resolve_feature_metas(
    feature_names: list[str], scope: str, scopes: list[str] | None = None
) -> dict[str, Any]:
    """Look up FeatureMetadata for each name, raising on missing entries.

    When *scopes* is provided, metadata is merged from all listed scopes.
    """
    from cogrid.core.component_registry import get_feature_types

    if scopes is not None:
        all_metas: list[Any] = []
        for s in scopes:
            all_metas.extend(get_feature_types(s))
    else:
        all_metas = get_feature_types(scope)
    meta_by_id = {m.feature_id: m for m in all_metas}

    for name in feature_names:
        if name not in meta_by_id:
            raise ValueError(f"Feature '{name}' not registered in scope '{scope}'.")
    return meta_by_id


def obs_dim_for_features(
    feature_names: list[str],
    scope: str,
    n_agents: int,
    scopes: list[str] | None = None,
    env_config: dict[str, Any] | None = None,
) -> int:
    """Compute total observation dimension for a list of feature names.

    Per-agent features contribute ``obs_dim * n_agents`` (one block per
    agent in ego-centric order). Global features contribute ``obs_dim`` once.

    When *env_config* is provided, each feature's ``compute_obs_dim`` is
    called instead of reading the frozen ``meta.obs_dim``.
    """
    meta_by_id = _resolve_feature_metas(feature_names, scope, scopes=scopes)

    total = 0
    for name in feature_names:
        meta = meta_by_id[name]
        dim = meta.cls.compute_obs_dim(scope, env_config)
        if meta.per_agent:
            if getattr(meta.cls, "focal_only", False):
                total += dim  # focal_only features are not duplicated for other agents
            else:
                total += dim * n_agents
        else:
            total += dim
    return total


def compose_feature_fns(
    feature_names: list[str],
    scope: str,
    n_agents: int,
    scopes: list[str] | None = None,
    preserve_order: bool = False,
    env_config: dict[str, Any] | None = None,
) -> Callable[[Any, int], ArrayLike]:
    """Compose registered features into a single ego-centric observation function.

    Concatenation order:

    1. Focal agent's per-agent features (including focal_only features)
    2. Other agents' per-agent features (ascending index, skipping focal;
       features with ``focal_only=True`` are excluded)
    3. Global features

    By default, names within each group are sorted alphabetically.
    Set ``preserve_order=True`` to keep the caller-provided order.

    When *env_config* is provided it is forwarded to ``build_feature_fn``
    for features whose signature accepts it.  If ``env_config`` contains
    ``observable_radius`` (not None), per-agent features receive a masked
    state view that hides cells beyond the agent's observable radius.

    Returns ``fn(state, agent_idx) -> (obs_dim,) float32``.
    """
    meta_by_id = _resolve_feature_metas(feature_names, scope, scopes=scopes)

    observable_radius = None
    if env_config is not None:
        observable_radius = env_config.get("observable_radius", None)

    # Separate per-agent and global
    if preserve_order:
        per_agent_names = [n for n in feature_names if meta_by_id[n].per_agent]
        global_names = [n for n in feature_names if not meta_by_id[n].per_agent]
    else:
        per_agent_names = sorted(n for n in feature_names if meta_by_id[n].per_agent)
        global_names = sorted(n for n in feature_names if not meta_by_id[n].per_agent)

    # Split per-agent features into regular and focal-only
    regular_pa_names = [
        n for n in per_agent_names if not getattr(meta_by_id[n].cls, "focal_only", False)
    ]
    focal_only_names = [
        n for n in per_agent_names if getattr(meta_by_id[n].cls, "focal_only", False)
    ]

    # Build feature functions once at compose time (not per call)
    regular_pa_fns: list[Callable[..., ArrayLike]] = [
        _call_build_feature_fn(meta_by_id[name].cls, scope, env_config) for name in regular_pa_names
    ]
    focal_only_fns: list[Callable[..., ArrayLike]] = [
        _call_build_feature_fn(meta_by_id[name].cls, scope, env_config) for name in focal_only_names
    ]
    global_fns: list[Callable[..., ArrayLike]] = [
        _call_build_feature_fn(meta_by_id[name].cls, scope, env_config) for name in global_names
    ]

    def composed_fn(state: Any, agent_idx: int) -> ArrayLike:
        parts: list[ArrayLike] = []

        # Apply masking for partial observability
        if observable_radius is not None:
            masked_state = mask_state_for_agent(state, agent_idx, observable_radius)
        else:
            masked_state = state

        # 1. Focal agent's per-agent features (regular + focal_only)
        for fn in regular_pa_fns:
            parts.append(fn(masked_state, agent_idx).ravel().astype(xp.float32))
        for fn in focal_only_fns:
            parts.append(fn(masked_state, agent_idx).ravel().astype(xp.float32))

        # 2. Other agents in ascending index order (skip focal)
        #    focal_only features are NOT included here
        for i in range(n_agents):
            if i == agent_idx:
                continue
            if observable_radius is not None:
                other_masked = mask_state_for_agent(state, i, observable_radius)
            else:
                other_masked = state
            for fn in regular_pa_fns:
                parts.append(fn(other_masked, i).ravel().astype(xp.float32))

        # 3. Global features
        for fn in global_fns:
            parts.append(fn(state).ravel().astype(xp.float32))

        return xp.concatenate(parts)

    return composed_fn


# ---------------------------------------------------------------------------
# Partial observability helpers
# ---------------------------------------------------------------------------


def mask_state_for_agent(state, agent_idx, radius):
    """Return a new StateView with state masked to agent's observable radius.

    Cells where ``manhattan_dist(agent_pos, cell) > radius`` are zeroed out
    in spatial maps.  Other agents outside the radius get position ``(-1, -1)``.
    Pot state outside the radius is zeroed.

    Parameters
    ----------
    state : StateView
        The full environment state.
    agent_idx : int
        Index of the focal agent.
    radius : int
        Manhattan distance radius for observability.

    Returns:
    -------
    StateView
        A new StateView with masked fields.
    """
    from cogrid.backend.state_view import StateView

    pos = state.agent_pos[agent_idx]  # (2,)
    H, W = state.object_type_map.shape

    # Build 2D manhattan distance mask
    rows = xp.arange(H, dtype=xp.int32)[:, None]
    cols = xp.arange(W, dtype=xp.int32)[None, :]
    dist = xp.abs(rows - pos[0]) + xp.abs(cols - pos[1])
    visible = dist <= radius

    # Mask spatial maps
    masked_otm = xp.where(visible, state.object_type_map, 0)
    masked_osm = xp.where(visible, state.object_state_map, 0)
    masked_wm = xp.where(visible, state.wall_map, 0)

    # Mask other agents' positions
    agent_dists = xp.abs(state.agent_pos[:, 0] - pos[0]) + xp.abs(state.agent_pos[:, 1] - pos[1])
    agent_visible = agent_dists <= radius
    masked_agent_pos = xp.where(agent_visible[:, None], state.agent_pos, -1)

    # Mask extra state (pots)
    masked_extra = dict(state.extra)
    if "pot_positions" in masked_extra:
        pot_pos = masked_extra["pot_positions"]
        pot_dists = xp.abs(pot_pos[:, 0] - pos[0]) + xp.abs(pot_pos[:, 1] - pos[1])
        pot_visible = pot_dists <= radius
        if "pot_contents" in masked_extra:
            masked_extra["pot_contents"] = xp.where(
                pot_visible[:, None], masked_extra["pot_contents"], 0
            )
        if "pot_timer" in masked_extra:
            masked_extra["pot_timer"] = xp.where(pot_visible, masked_extra["pot_timer"], 0)

    return StateView(
        agent_pos=masked_agent_pos,
        agent_dir=state.agent_dir,
        agent_inv=state.agent_inv,
        wall_map=masked_wm,
        object_type_map=masked_otm,
        object_state_map=masked_osm,
        extra=masked_extra,
    )
