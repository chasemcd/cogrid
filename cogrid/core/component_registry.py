"""Component metadata registry for GridObject and Reward type registration.

Stores metadata about GridObject subclasses (discovered classmethods, static
properties) and ArrayReward subclasses. Populated at import time by the
``@register_object_type`` decorator and ``@register_reward_type`` decorator.

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

    @property
    def has_tick(self) -> bool:
        return "build_tick_fn" in self.methods

    @property
    def has_interaction(self) -> bool:
        return "build_interaction_fn" in self.methods

    @property
    def has_extra_state(self) -> bool:
        return "extra_state_schema" in self.methods


@dataclass(frozen=True)
class RewardMetadata:
    """Metadata for a registered ArrayReward subclass."""

    scope: str
    reward_id: str
    cls: type
    default_coefficient: float
    default_common_reward: bool


# ---------------------------------------------------------------------------
# Registries (private)
# ---------------------------------------------------------------------------

_COMPONENT_METADATA: dict[tuple[str, str], ComponentMetadata] = {}
_REWARD_TYPE_REGISTRY: dict[tuple[str, str], RewardMetadata] = {}


# ---------------------------------------------------------------------------
# Signature validation helpers (private)
# ---------------------------------------------------------------------------

# All four component classmethods take only ``cls`` -- no additional params.
_EXPECTED_SIGNATURES: dict[str, list[str]] = {
    "build_tick_fn": [],
    "build_interaction_fn": [],
    "extra_state_schema": [],
    "extra_state_builder": [],
}

_EXPECTED_REWARD_COMPUTE_PARAMS = ["prev_state", "state", "actions", "reward_config"]


def _validate_classmethod_signature(cls: type, method_name: str, method: Any) -> None:
    """Validate that *method* on *cls* matches the expected signature.

    For classmethods the ``cls`` parameter is auto-stripped by
    ``inspect.signature``.  If an author accidentally defines an instance
    method (``def build_tick_fn(self)``), ``self`` will appear in the
    params list and the check will fail with a clear message.
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


def _validate_reward_compute_signature(cls: type) -> None:
    """Validate that ``cls.compute`` has the expected signature.

    Expected: ``(self, prev_state, state, actions, reward_config)``.
    """
    sig = inspect.signature(cls.compute)
    params = list(sig.parameters.keys())

    # Strip leading ``self``
    if params and params[0] == "self":
        params = params[1:]

    if params != _EXPECTED_REWARD_COMPUTE_PARAMS:
        raise TypeError(
            f"{cls.__name__}.compute() has params {params} (after self), "
            f"expected {_EXPECTED_REWARD_COMPUTE_PARAMS}."
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
    )
    _COMPONENT_METADATA[(scope, object_id)] = meta


def register_reward_type(
    reward_id: str,
    scope: str = "global",
    coefficient: float = 1.0,
    common_reward: bool = False,
):
    """Decorator that registers an ArrayReward subclass.

    Usage::

        @register_reward_type("delivery", scope="overcooked", coefficient=20.0)
        class DeliveryReward(ArrayReward):
            def compute(self, prev_state, state, actions, reward_config):
                ...
    """

    def decorator(cls):
        # Must have a compute method
        if not hasattr(cls, "compute") or not callable(getattr(cls, "compute")):
            raise TypeError(
                f"{cls.__name__} must define a callable 'compute' method "
                f"to be registered as a reward type."
            )

        _validate_reward_compute_signature(cls)

        key = (scope, reward_id)
        if key in _REWARD_TYPE_REGISTRY:
            existing = _REWARD_TYPE_REGISTRY[key]
            raise ValueError(
                f"Duplicate reward type '{reward_id}' in scope '{scope}': "
                f"{existing.cls.__name__} and {cls.__name__}"
            )

        _REWARD_TYPE_REGISTRY[key] = RewardMetadata(
            scope=scope,
            reward_id=reward_id,
            cls=cls,
            default_coefficient=coefficient,
            default_common_reward=common_reward,
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


def get_reward_types(scope: str = "global") -> list[RewardMetadata]:
    """Return all RewardMetadata entries for *scope*, sorted by reward_id."""
    return sorted(
        [m for m in _REWARD_TYPE_REGISTRY.values() if m.scope == scope],
        key=lambda m: m.reward_id,
    )


def get_tickable_components(scope: str = "global") -> list[ComponentMetadata]:
    """Return components that have a ``build_tick_fn`` classmethod."""
    return [m for m in get_all_components(scope) if m.has_tick]


def get_components_with_extra_state(scope: str = "global") -> list[ComponentMetadata]:
    """Return components that have an ``extra_state_schema`` classmethod."""
    return [m for m in get_all_components(scope) if m.has_extra_state]
