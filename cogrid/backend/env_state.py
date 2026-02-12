"""Immutable environment state container.

Defines :class:`EnvState`, a frozen dataclass that bundles all environment
state arrays into a single object.  On the numpy backend it is a plain
data container; on the JAX backend it is registered as a pytree so it can
flow through ``jax.jit``, ``jax.vmap``, etc.

State transitions use ``dataclasses.replace``::

    new_state = dataclasses.replace(state, agent_pos=new_pos)

Direct attribute assignment is forbidden (frozen=True).

Usage::

    from cogrid.backend.env_state import EnvState, create_env_state

    state = create_env_state(
        agent_pos=np.zeros((2, 2), dtype=np.int32),
        ...
    )
"""

from __future__ import annotations

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
        pot_contents:     (n_pots, 3) int32   -- pot ingredient type IDs, -1 = empty
        pot_timer:        (n_pots,) int32     -- cooking timer per pot
        pot_positions:    (n_pots, 2) int32   -- pot row, col positions
        rng_key:          (2,) uint32 or None -- JAX PRNG key (None on numpy backend)
        time:             () int32            -- scalar timestep

    Static fields (compile-time constants, not traced):
        n_agents:   int -- number of agents
        height:     int -- grid height
        width:      int -- grid width
        n_pots:     int -- number of pots
        action_set: str -- "cardinal" or "rotation"
    """

    # --- Dynamic fields (JAX arrays when on jax backend, numpy otherwise) ---
    agent_pos: object          # (n_agents, 2) int32
    agent_dir: object          # (n_agents,) int32
    agent_inv: object          # (n_agents, 1) int32, -1 sentinel for empty
    wall_map: object           # (H, W) int32
    object_type_map: object    # (H, W) int32
    object_state_map: object   # (H, W) int32
    pot_contents: object       # (n_pots, 3) int32, -1 sentinel
    pot_timer: object          # (n_pots,) int32
    pot_positions: object      # (n_pots, 2) int32
    rng_key: object            # (2,) uint32 JAX PRNG key, or None on numpy
    time: object               # () int32 scalar timestep

    # --- Static fields (compile-time constants, not traced) ---
    n_agents: int = field(metadata=dict(static=True), default=2)
    height: int = field(metadata=dict(static=True), default=7)
    width: int = field(metadata=dict(static=True), default=7)
    n_pots: int = field(metadata=dict(static=True), default=0)
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
        pot_contents=np.full((1, 3), -1, dtype=np.int32),
        pot_timer=np.zeros(1, dtype=np.int32),
        pot_positions=np.zeros((1, 2), dtype=np.int32),
        rng_key=None,
        time=np.int32(0),
        n_agents=2,
        height=7,
        width=7,
        n_pots=1,
        action_set="cardinal",
    )
    print(f"EnvState created: {state.n_agents} agents, {state.height}x{state.width} grid")
