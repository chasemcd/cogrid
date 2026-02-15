"""ArrayFeature base class for array-based feature components.

Each feature is a subclass with class attributes ``per_agent`` and ``obs_dim``,
and a ``build_feature_fn`` classmethod that returns a pure function.

Per-agent features: ``fn(state_dict, agent_idx) -> ndarray``
Global features: ``fn(state_dict) -> ndarray``

Feature composition is handled by ``compose_feature_fns()`` in this module.

Environment-specific features live in their respective envs/ modules.
"""

# Re-export for convenience (decorator lives in component_registry)
from cogrid.backend import xp
from cogrid.core.component_registry import register_feature_type  # noqa: F401


class ArrayFeature:
    """Base class for array-based feature extractors.

    Subclasses MUST define:
        - ``per_agent``: bool class attribute. True if feature is per-agent,
          False if global.
        - ``obs_dim``: int class attribute. Dimensionality of the feature
          output after ravel().
        - ``build_feature_fn(cls, scope)``: classmethod that returns a pure
          function. For per_agent=True: fn(state_dict, agent_idx) -> ndarray.
          For per_agent=False: fn(state_dict) -> ndarray.

    ``state_dict`` is a :class:`~cogrid.backend.state_view.StateView` —
    a frozen dataclass with dot access for core fields (``agent_pos``,
    ``agent_dir``, etc.) and ``__getattr__`` fallthrough for extras.

    Usage::

        @register_feature_type("agent_dir", scope="global")
        class AgentDir(ArrayFeature):
            per_agent = True
            obs_dim = 4

            @classmethod
            def build_feature_fn(cls, scope):
                def fn(state_dict, agent_idx):
                    from cogrid.backend import xp
                    return (xp.arange(4) == state_dict.agent_dir[agent_idx]).astype(xp.int32)
                return fn
    """

    per_agent: bool
    obs_dim: int

    @classmethod
    def build_feature_fn(cls, scope):
        """Build and return the feature extraction function.

        Args:
            scope: Registry scope name for pre-computing type IDs
                and scope-dependent lookups.

        Returns:
            For per_agent=True: fn(state_dict, agent_idx) -> ndarray
            For per_agent=False: fn(state_dict) -> ndarray
        """
        raise NotImplementedError(
            f"{cls.__name__}.build_feature_fn() is not implemented. "
            f"Subclasses must override build_feature_fn()."
        )


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def _resolve_feature_metas(feature_names, scope, scopes=None):
    """Look up FeatureMetadata for each name, raising on missing entries.

    Args:
        feature_names: List of feature name strings to resolve.
        scope: Default registry scope (used when *scopes* is None).
        scopes: Optional list of scope strings.  When provided, metadata
            is merged from ALL listed scopes, allowing cross-scope lookup.

    Returns:
        dict mapping feature_id -> FeatureMetadata
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
            raise ValueError(
                f"Feature '{name}' not registered in scope '{scope}'."
            )
    return meta_by_id


def obs_dim_for_features(feature_names, scope, n_agents, scopes=None):
    """Compute total observation dimension for a list of feature names.

    Per-agent features contribute ``obs_dim * n_agents`` (one block per agent
    in ego-centric order).  Global features contribute ``obs_dim`` once.

    Args:
        feature_names: List of registered feature name strings.
        scope: Registry scope.
        n_agents: Number of agents.
        scopes: Optional list of scope strings for multi-scope lookup.

    Returns:
        int: Total observation dimension.
    """
    meta_by_id = _resolve_feature_metas(feature_names, scope, scopes=scopes)

    total = 0
    for name in feature_names:
        meta = meta_by_id[name]
        if meta.per_agent:
            total += meta.obs_dim * n_agents
        else:
            total += meta.obs_dim
    return total


def compose_feature_fns(feature_names, scope, n_agents, scopes=None, preserve_order=False):
    """Compose registered features into a single ego-centric observation function.

    Discovers features by name from the registry for the given *scope* (or
    multiple *scopes*).  Builds each feature's function via
    ``build_feature_fn(scope)``.  Returns a single function that concatenates
    all features in ego-centric order.  ``state_dict`` is a
    :class:`~cogrid.backend.state_view.StateView`.

    1. Focal agent's per-agent features
    2. Other agents' per-agent features in ascending index order (skipping focal),
       each agent's features in the same order
    3. Global features

    When ``preserve_order`` is False (default), per-agent and global name lists
    are sorted alphabetically.  When True, the caller-provided order within
    ``feature_names`` is preserved (per-agent and global groups are extracted
    in list order).

    All feature outputs are raveled and cast to float32 before concatenation.

    Args:
        feature_names: List of registered feature name strings.
        scope: Registry scope (used as default when *scopes* is None, and
            as the scope passed to ``build_feature_fn``).
        n_agents: Number of agents.
        scopes: Optional list of scope strings for multi-scope lookup.
        preserve_order: If True, keep the caller-provided feature order
            instead of sorting alphabetically.

    Returns:
        fn(state_dict, agent_idx) -> (obs_dim,) float32 ndarray
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
        meta_by_id[name].cls.build_feature_fn(scope) for name in per_agent_names
    ]
    global_fns = [
        meta_by_id[name].cls.build_feature_fn(scope) for name in global_names
    ]

    def composed_fn(state_dict, agent_idx):
        parts = []

        # 1. Focal agent's per-agent features
        for fn in per_agent_fns:
            parts.append(fn(state_dict, agent_idx).ravel().astype(xp.float32))

        # 2. Other agents in ascending index order (skip focal)
        for i in range(n_agents):
            if i == agent_idx:
                continue
            for fn in per_agent_fns:
                parts.append(fn(state_dict, i).ravel().astype(xp.float32))

        # 3. Global features
        for fn in global_fns:
            parts.append(fn(state_dict).ravel().astype(xp.float32))

        return xp.concatenate(parts)

    return composed_fn
