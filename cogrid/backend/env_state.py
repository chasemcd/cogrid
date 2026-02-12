"""Immutable environment state container.

Defines :class:`EnvState`, a frozen dataclass that bundles all environment
state arrays into a single object.  On the numpy backend it is a plain
data container; on the JAX backend it is registered as a pytree so it can
flow through ``jax.jit``, ``jax.vmap``, etc.

State transitions use ``dataclasses.replace``::

    new_state = dataclasses.replace(state, agent_pos=new_pos)

Direct attribute assignment is forbidden (frozen=True).

The ``extra_state`` field holds a dict of scope-specific arrays keyed by
scope-prefixed strings (e.g. ``"overcooked.pot_contents"``).  This
replaces the former pot-specific fields and allows any environment to
attach its own state arrays without modifying the core dataclass.

Access helpers::

    from cogrid.backend.env_state import get_extra, replace_extra

    timer = get_extra(state, "pot_timer", scope="overcooked")
    state = replace_extra(state, "pot_timer", new_timer, scope="overcooked")

Usage::

    from cogrid.backend.env_state import EnvState, create_env_state

    state = create_env_state(
        agent_pos=np.zeros((2, 2), dtype=np.int32),
        ...
        extra_state={"overcooked.pot_timer": np.zeros(2, dtype=np.int32)},
    )
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

# NOTE: No JAX imports at module level -- the class must work without JAX.

_pytree_registered: bool = False


@dataclass(frozen=True)
class EnvState:
    """Immutable environment state.

    All dynamic fields are fixed-shape arrays (numpy or JAX depending on
    the active backend).  Static fields are Python scalars used as
    compile-time constants under JIT (JAX recompiles for each distinct
    combination of static values).

    Dynamic fields (traced through JIT):
        agent_pos:        (n_agents, 2) int32 -- row, col positions
        agent_dir:        (n_agents,) int32   -- direction enum values
        agent_inv:        (n_agents, 1) int32 -- held item type IDs, -1 = empty
        wall_map:         (H, W) int32        -- 1 where walls exist
        object_type_map:  (H, W) int32        -- object type IDs per cell
        object_state_map: (H, W) int32        -- object state per cell
        extra_state:      dict[str, array]    -- scope-prefixed env-specific arrays
        rng_key:          (2,) uint32 or None -- JAX PRNG key (None on numpy backend)
        time:             () int32            -- scalar timestep

    Static fields (compile-time constants, not traced):
        n_agents:   int -- number of agents
        height:     int -- grid height
        width:      int -- grid width
        action_set: str -- "cardinal" or "rotation"
    """

    # --- Dynamic fields (JAX arrays when on jax backend, numpy otherwise) ---
    agent_pos: object          # (n_agents, 2) int32
    agent_dir: object          # (n_agents,) int32
    agent_inv: object          # (n_agents, 1) int32, -1 sentinel for empty
    wall_map: object           # (H, W) int32
    object_type_map: object    # (H, W) int32
    object_state_map: object   # (H, W) int32
    extra_state: object        # dict[str, array], scope-prefixed keys
    rng_key: object            # (2,) uint32 JAX PRNG key, or None on numpy
    time: object               # () int32 scalar timestep

    # --- Static fields (compile-time constants, not traced) ---
    n_agents: int = field(metadata=dict(static=True), default=2)
    height: int = field(metadata=dict(static=True), default=7)
    width: int = field(metadata=dict(static=True), default=7)
    action_set: str = field(metadata=dict(static=True), default="cardinal")


def register_envstate_pytree() -> None:
    """Register :class:`EnvState` as a JAX pytree node.

    Imports ``jax.tree_util.register_dataclass`` and calls it on
    :class:`EnvState`.  Safe to call multiple times (idempotent).
    Must only be called when the JAX backend is active.
    """
    global _pytree_registered
    if _pytree_registered:
        return

    import jax.tree_util
    jax.tree_util.register_dataclass(EnvState)
    _pytree_registered = True


def create_env_state(**kwargs) -> EnvState:
    """Create an :class:`EnvState` from keyword arguments.

    If the active backend is ``'jax'``, calls
    :func:`register_envstate_pytree` first to ensure the pytree
    registration is in place.

    All keyword arguments are forwarded to the :class:`EnvState`
    constructor.

    Returns:
        A new :class:`EnvState` instance.
    """
    from cogrid.backend import get_backend

    if get_backend() == "jax":
        register_envstate_pytree()

    return EnvState(**kwargs)


# ======================================================================
# extra_state helpers
# ======================================================================


def get_extra(state, key, scope=None):
    """Get a value from state.extra_state with optional scope prefixing."""
    full_key = f"{scope}.{key}" if scope else key
    if full_key not in state.extra_state:
        raise KeyError(
            f"extra_state key '{full_key}' not found. "
            f"Available: {list(state.extra_state.keys())}"
        )
    return state.extra_state[full_key]


def replace_extra(state, key, value, scope=None):
    """Return new EnvState with one extra_state value replaced."""
    full_key = f"{scope}.{key}" if scope else key
    new_extra = {**state.extra_state, full_key: value}
    return dataclasses.replace(state, extra_state=new_extra)


def validate_extra_state(extra_state, schema):
    """Validate extra_state dict against a schema.

    Schema is a dict mapping key -> {"shape": tuple, "dtype": str}.
    Shape tuples may contain symbolic dims (strings) which are checked
    for dimensional consistency only (not exact value).

    Raises ValueError if validation fails.
    """
    for key, spec in schema.items():
        if key not in extra_state:
            raise ValueError(f"Missing required extra_state key: '{key}'")
        arr = extra_state[key]
        expected_ndim = len(spec["shape"])
        if len(arr.shape) != expected_ndim:
            raise ValueError(
                f"extra_state['{key}'] has {len(arr.shape)} dims, "
                f"expected {expected_ndim}"
            )


if __name__ == "__main__":
    # Quick smoke test -- numpy only, no JAX dependency
    import numpy as np

    state = EnvState(
        agent_pos=np.zeros((2, 2), dtype=np.int32),
        agent_dir=np.zeros(2, dtype=np.int32),
        agent_inv=np.full((2, 1), -1, dtype=np.int32),
        wall_map=np.zeros((7, 7), dtype=np.int32),
        object_type_map=np.zeros((7, 7), dtype=np.int32),
        object_state_map=np.zeros((7, 7), dtype=np.int32),
        extra_state={
            "overcooked.pot_contents": np.full((1, 3), -1, dtype=np.int32),
            "overcooked.pot_timer": np.zeros(1, dtype=np.int32),
            "overcooked.pot_positions": np.zeros((1, 2), dtype=np.int32),
        },
        rng_key=None,
        time=np.int32(0),
        n_agents=2,
        height=7,
        width=7,
        action_set="cardinal",
    )
    print(f"EnvState created: {state.n_agents} agents, {state.height}x{state.width} grid")

    # Test helpers
    timer = get_extra(state, "pot_timer", scope="overcooked")
    print(f"pot_timer shape: {timer.shape}")

    state2 = replace_extra(state, "pot_timer", np.ones(1, dtype=np.int32), scope="overcooked")
    print(f"replaced timer value: {get_extra(state2, 'pot_timer', scope='overcooked')}")

    # Test validate
    schema = {
        "overcooked.pot_contents": {"shape": ("n_pots", 3), "dtype": "int32"},
        "overcooked.pot_timer": {"shape": ("n_pots",), "dtype": "int32"},
        "overcooked.pot_positions": {"shape": ("n_pots", 2), "dtype": "int32"},
    }
    validate_extra_state(state.extra_state, schema)
    print("validation passed")
