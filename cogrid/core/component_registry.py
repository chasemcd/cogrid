"""Component metadata registry for GridObject and Feature type registration.

Stores metadata about GridObject subclasses (discovered classmethods, static
properties) and Feature subclasses. Populated at import time by the
``@register_object_type`` and ``@register_feature_type`` decorators.

This module must NOT import from ``cogrid.core.grid_object`` at module level
to avoid circular imports. The decorator in grid_object.py uses a lazy import
to call into this module.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentMetadata:
    """Metadata for a registered GridObject subclass."""

    scope: str
    object_id: str
    cls: type
    char: str
    properties: dict[str, bool]
    methods: dict[str, Any] = field(default_factory=dict)
    container_meta: dict | None = None

    @property
    def has_tick(self) -> bool:
        """True if this component provides a tick function."""
        return "build_tick_fn" in self.methods

    @property
    def has_extra_state(self) -> bool:
        """True if this component defines extra state schema."""
        return "extra_state_schema" in self.methods

    @property
    def has_static_tables(self) -> bool:
        """True if this component provides static lookup tables."""
        return "build_static_tables" in self.methods

    @property
    def has_render_sync(self) -> bool:
        """True if this component provides a render sync function."""
        return "build_render_sync_fn" in self.methods

    @property
    def has_container(self) -> bool:
        """True if this component declares a Container descriptor."""
        return self.container_meta is not None


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata for a registered Feature subclass."""

    scope: str
    feature_id: str
    cls: type
    per_agent: bool
    obs_dim: int


# ---------------------------------------------------------------------------
# Registries (private)
# ---------------------------------------------------------------------------

_COMPONENT_METADATA: dict[tuple[str, str], ComponentMetadata] = {}
_FEATURE_TYPE_REGISTRY: dict[tuple[str, str], FeatureMetadata] = {}
_PRE_COMPOSE_HOOKS: dict[str, callable] = {}
_LAYOUT_INDEX_REGISTRY: dict[str, dict[str, int]] = {}


# ---------------------------------------------------------------------------
# Signature validation helpers (private)
# ---------------------------------------------------------------------------

# All four component classmethods take only ``cls`` -- no additional params.
_EXPECTED_SIGNATURES: dict[str, list[str]] = {
    "build_tick_fn": [],
    "extra_state_schema": [],
    "extra_state_builder": [],
    "build_static_tables": [],
    "build_render_sync_fn": [],
}

def _validate_classmethod_signature(cls: type, method_name: str, method: Any) -> None:
    """Validate that *method* matches expected params.

    Catches instance methods mistyped as classmethods (``self`` in params).
    """
    expected = _EXPECTED_SIGNATURES.get(method_name)
    if expected is None:
        return  # unknown method -- skip validation

    sig = inspect.signature(method)
    # For a real @classmethod, inspect.signature auto-strips ``cls``.
    # If an author writes an instance method instead, ``self`` will remain
    # in the params list and will cause a mismatch against expected=[].
    actual = list(sig.parameters.keys())

    if actual != expected:
        raise TypeError(
            f"{cls.__name__}.{method_name}() has params {actual}, "
            f"expected {expected}. "
            f"Ensure it is a @classmethod with signature "
            f"def {method_name}(cls)."
        )


# ---------------------------------------------------------------------------
# Registration functions
# ---------------------------------------------------------------------------


def register_component_metadata(
    scope: str,
    object_id: str,
    cls: type,
    properties: dict[str, bool],
    methods: dict[str, Any],
    container_meta: dict | None = None,
) -> None:
    """Store :class:`ComponentMetadata` for a GridObject subclass.

    Called by the ``@register_object_type`` decorator. This function only
    stores metadata -- it does NOT validate completeness (e.g. schema
    without builder is allowed).
    """
    meta = ComponentMetadata(
        scope=scope,
        object_id=object_id,
        cls=cls,
        char=cls.char,
        properties=properties,
        methods=methods,
        container_meta=container_meta,
    )
    _COMPONENT_METADATA[(scope, object_id)] = meta


def register_feature_type(feature_id: str, scope: str = "global"):
    """Decorator that registers a Feature subclass.

    Usage::

        @register_feature_type("agent_dir", scope="global")
        class AgentDir(Feature):
            per_agent = True
            obs_dim = 4

            @classmethod
            def build_feature_fn(cls, scope): ...
    """

    def decorator(cls):
        # Validate cls has per_agent (bool) and obs_dim (int) class attributes
        if not hasattr(cls, "per_agent") or not isinstance(cls.per_agent, bool):
            raise TypeError(f"{cls.__name__} must define 'per_agent' as a bool class attribute.")
        if not hasattr(cls, "obs_dim") or not isinstance(cls.obs_dim, int):
            raise TypeError(f"{cls.__name__} must define 'obs_dim' as an int class attribute.")

        # Validate build_feature_fn exists and has correct signature
        if not hasattr(cls, "build_feature_fn") or not callable(getattr(cls, "build_feature_fn")):
            raise TypeError(
                f"{cls.__name__} must define a callable 'build_feature_fn' classmethod."
            )

        # Feature.build_feature_fn takes (cls, scope) -- validate the
        # scope parameter directly (not via _EXPECTED_SIGNATURES which holds
        # the old GridObject convention of no params).
        sig = inspect.signature(cls.build_feature_fn)
        actual = list(sig.parameters.keys())
        if actual not in (["scope"], ["scope", "env_config"]):
            raise TypeError(
                f"{cls.__name__}.build_feature_fn() has params {actual}, "
                f"expected ['scope'] or ['scope', 'env_config']. "
                f"Ensure it is a @classmethod with signature "
                f"def build_feature_fn(cls, scope) or "
                f"def build_feature_fn(cls, scope, env_config=None)."
            )

        # Check for duplicate registration
        key = (scope, feature_id)
        if key in _FEATURE_TYPE_REGISTRY:
            existing = _FEATURE_TYPE_REGISTRY[key]
            same_class = existing.cls.__name__ == cls.__name__ and getattr(
                existing.cls, "__module__", None
            ) == getattr(cls, "__module__", None)
            if not same_class:
                raise ValueError(
                    f"Duplicate feature type '{feature_id}' in scope '{scope}': "
                    f"{existing.cls.__name__} and {cls.__name__}"
                )

        _FEATURE_TYPE_REGISTRY[key] = FeatureMetadata(
            scope=scope,
            feature_id=feature_id,
            cls=cls,
            per_agent=cls.per_agent,
            obs_dim=cls.obs_dim,
        )
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------


def get_component_metadata(object_id: str, scope: str = "global") -> ComponentMetadata | None:
    """Look up component metadata by (scope, object_id). Returns None if not found."""
    return _COMPONENT_METADATA.get((scope, object_id))


def get_all_components(scope: str = "global") -> list[ComponentMetadata]:
    """Return all ComponentMetadata entries for *scope*, sorted by object_id."""
    return sorted(
        [m for m in _COMPONENT_METADATA.values() if m.scope == scope],
        key=lambda m: m.object_id,
    )


def get_feature_types(scope: str = "global") -> list[FeatureMetadata]:
    """Return all FeatureMetadata entries for *scope*, sorted by feature_id."""
    return sorted(
        [m for m in _FEATURE_TYPE_REGISTRY.values() if m.scope == scope],
        key=lambda m: m.feature_id,
    )


def get_tickable_components(scope: str = "global") -> list[ComponentMetadata]:
    """Return components that have a ``build_tick_fn`` classmethod."""
    return [m for m in get_all_components(scope) if m.has_tick]


def get_components_with_extra_state(scope: str = "global") -> list[ComponentMetadata]:
    """Return components that have an ``extra_state_schema`` classmethod."""
    return [m for m in get_all_components(scope) if m.has_extra_state]


def get_container_components(scope: str = "global") -> list[ComponentMetadata]:
    """Return components that declare a ``Container`` descriptor."""
    return [m for m in get_all_components(scope) if m.has_container]


# ---------------------------------------------------------------------------
# Pre-compose hook registration
# ---------------------------------------------------------------------------


def register_pre_compose_hook(scope: str, hook: callable) -> None:
    """Register a hook called before feature composition.

    ``hook(layout_idx: int, scope: str, env_config: dict | None = None) -> None``
    """
    _PRE_COMPOSE_HOOKS[scope] = hook


def get_pre_compose_hook(scope: str) -> callable | None:
    """Return the registered pre-compose hook for *scope*, or None."""
    return _PRE_COMPOSE_HOOKS.get(scope)


# ---------------------------------------------------------------------------
# Layout index registration
# ---------------------------------------------------------------------------


def register_layout_indices(scope: str, layout_map: dict[str, int]) -> None:
    """Register a layout-name-to-index mapping for a scope."""
    _LAYOUT_INDEX_REGISTRY[scope] = dict(layout_map)


def get_layout_index(scope: str, layout_id: str | None) -> int:
    """Return the integer index for *layout_id* in *scope*, or 0.

    Returns 0 if *layout_id* is None, *scope* has no registered mapping,
    or *layout_id* is not found in the mapping.
    """
    if layout_id is None:
        return 0
    mapping = _LAYOUT_INDEX_REGISTRY.get(scope)
    if mapping is None:
        return 0
    return mapping.get(layout_id, 0)
