"""Feature base class for feature components.

Each feature is a subclass with class attributes ``per_agent`` and ``obs_dim``,
and a ``build_feature_fn`` classmethod that returns a pure function.

Per-agent features: ``fn(state, agent_idx) -> ndarray``
Global features: ``fn(state) -> ndarray``

Feature composition is handled by ``compose_feature_fns()`` in this module.

Environment-specific features live in their respective envs/ modules.
"""

import inspect

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
    def compute_obs_dim(cls, scope, env_config=None):
        """Compute observation dimension, optionally using env_config.

        Subclasses may override to return a config-dependent dimension.
        Default returns the static ``cls.obs_dim``.
        """
        return cls.obs_dim

    @classmethod
    def build_feature_fn(cls, scope):
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


def _call_build_feature_fn(cls, scope, env_config):
    """Call cls.build_feature_fn, passing env_config if the method accepts it."""
    sig = inspect.signature(cls.build_feature_fn)
    if "env_config" in sig.parameters:
        return cls.build_feature_fn(scope, env_config=env_config)
    return cls.build_feature_fn(scope)


def _resolve_feature_metas(feature_names, scope, scopes=None):
    """Look up FeatureMetadata for each name, raising on missing entries.

    When *scopes* is provided, metadata is merged from all listed scopes.
    """
    from cogrid.core.component_registry import get_feature_types

    if scopes is not None:
        all_metas = []
        for s in scopes:
            all_metas.extend(get_feature_types(s))
    else:
        all_metas = get_feature_types(scope)
    meta_by_id = {m.feature_id: m for m in all_metas}

    for name in feature_names:
        if name not in meta_by_id:
            raise ValueError(f"Feature '{name}' not registered in scope '{scope}'.")
    return meta_by_id


def obs_dim_for_features(feature_names, scope, n_agents, scopes=None, env_config=None):
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
            total += dim * n_agents
        else:
            total += dim
    return total


def compose_feature_fns(
    feature_names, scope, n_agents, scopes=None, preserve_order=False, env_config=None
):
    """Compose registered features into a single ego-centric observation function.

    Concatenation order:

    1. Focal agent's per-agent features
    2. Other agents' per-agent features (ascending index, skipping focal)
    3. Global features

    By default, names within each group are sorted alphabetically.
    Set ``preserve_order=True`` to keep the caller-provided order.

    When *env_config* is provided it is forwarded to ``build_feature_fn``
    for features whose signature accepts it.

    Returns ``fn(state, agent_idx) -> (obs_dim,) float32``.
    """
    meta_by_id = _resolve_feature_metas(feature_names, scope, scopes=scopes)

    # Separate per-agent and global
    if preserve_order:
        per_agent_names = [n for n in feature_names if meta_by_id[n].per_agent]
        global_names = [n for n in feature_names if not meta_by_id[n].per_agent]
    else:
        per_agent_names = sorted(n for n in feature_names if meta_by_id[n].per_agent)
        global_names = sorted(n for n in feature_names if not meta_by_id[n].per_agent)

    # Build feature functions once at compose time (not per call)
    per_agent_fns = [
        _call_build_feature_fn(meta_by_id[name].cls, scope, env_config) for name in per_agent_names
    ]
    global_fns = [
        _call_build_feature_fn(meta_by_id[name].cls, scope, env_config) for name in global_names
    ]

    def composed_fn(state, agent_idx):
        parts = []

        # 1. Focal agent's per-agent features
        for fn in per_agent_fns:
            parts.append(fn(state, agent_idx).ravel().astype(xp.float32))

        # 2. Other agents in ascending index order (skip focal)
        for i in range(n_agents):
            if i == agent_idx:
                continue
            for fn in per_agent_fns:
                parts.append(fn(state, i).ravel().astype(xp.float32))

        # 3. Global features
        for fn in global_fns:
            parts.append(fn(state).ravel().astype(xp.float32))

        return xp.concatenate(parts)

    return composed_fn
