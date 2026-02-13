"""ArrayFeature base class for array-based feature components.

Each feature is a subclass with class attributes ``per_agent`` and ``obs_dim``,
and a ``build_feature_fn`` classmethod that returns a pure function.

Per-agent features: ``fn(state_dict, agent_idx) -> ndarray``
Global features: ``fn(state_dict) -> ndarray``

Feature composition is handled by ``compose_feature_fns()`` in this module.

Environment-specific features live in their respective envs/ modules.
"""

# Re-export for convenience (decorator lives in component_registry)
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

    Usage::

        @register_feature_type("agent_dir", scope="global")
        class AgentDir(ArrayFeature):
            per_agent = True
            obs_dim = 4

            @classmethod
            def build_feature_fn(cls, scope):
                def fn(state_dict, agent_idx):
                    from cogrid.backend import xp
                    return (xp.arange(4) == state_dict["agent_dir"][agent_idx]).astype(xp.int32)
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
