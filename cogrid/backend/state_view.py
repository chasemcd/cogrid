"""Dot-accessible view of environment state arrays.

:class:`StateView` is a frozen dataclass that replaces the plain ``dict``
returned by :func:`~cogrid.core.step_pipeline.envstate_to_dict`.  It
provides IDE-friendly dot access for the six core fields that every
feature / reward / termination function uses, plus ``__getattr__``
fallthrough to an ``extra`` dict for scope-specific arrays (e.g.
``state.pot_timer``).

On the JAX backend the class is registered as a pytree so it flows
through ``jax.jit`` and ``jax.vmap`` without issue.

Usage::

    state = StateView(
        agent_pos=..., agent_dir=..., agent_inv=...,
        wall_map=..., object_type_map=..., object_state_map=...,
        extra={"pot_timer": ...},
    )
    state.agent_pos       # core field  – full autocomplete
    state.pot_timer       # extra field – resolved via __getattr__
    state.nonexistent     # raises AttributeError
"""

from __future__ import annotations

from dataclasses import dataclass, field


_stateview_pytree_registered: bool = False


@dataclass(frozen=True)
class StateView:
    """Immutable, dot-accessible view of environment state arrays.

    Core fields (IDE-discoverable):
        agent_pos:        (n_agents, 2) int32
        agent_dir:        (n_agents,)   int32
        agent_inv:        (n_agents, 1) int32
        wall_map:         (H, W) int32
        object_type_map:  (H, W) int32
        object_state_map: (H, W) int32

    Extra fields (scope-specific, resolved via __getattr__):
        Stored in ``extra`` dict.  Access as ``state.<key>``.
    """

    agent_pos: object        # (n_agents, 2) int32
    agent_dir: object        # (n_agents,) int32
    agent_inv: object        # (n_agents, 1) int32
    wall_map: object         # (H, W) int32
    object_type_map: object  # (H, W) int32
    object_state_map: object # (H, W) int32
    extra: dict = field(default_factory=dict)

    def __getattr__(self, name):
        # Only called for attributes not found by normal lookup.
        # Frozen dataclass stores fields in __dict__, so core fields
        # never reach here.
        try:
            return self.extra[name]
        except KeyError:
            raise AttributeError(f"StateView has no field '{name}'")


def register_stateview_pytree() -> None:
    """Register :class:`StateView` as a JAX pytree node (idempotent)."""
    global _stateview_pytree_registered
    if _stateview_pytree_registered:
        return

    import jax.tree_util
    jax.tree_util.register_dataclass(StateView)
    _stateview_pytree_registered = True
