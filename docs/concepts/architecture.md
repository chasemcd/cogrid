# Architecture

This page explains the key architectural concepts in CoGrid: the array backend
system, the immutable state container, the component registration API, and the
step pipeline that ties everything together.

## Overview

CoGrid separates the stateful PettingZoo wrapper from a pure functional
simulation core. All environment logic is composed from registered components
at initialization time, then executed as a single compiled pipeline at runtime.

```text
+-----------------------------+
|        CoGridEnv            |   Stateful wrapper (PettingZoo API)
+-----------------------------+
| config dict                 |   Declares scope, features, rewards, layout
+----+------------------------+
     |
     v
+----+------------------------+
|      Autowire Layer         |   Composes registered components into
|  build_scope_config()       |   a scope_config dict at init time
|  build_reward_config()      |
|  build_feature_config()     |
+----+------------------------+
     |
     v
+----+------------------------+
|      Step Pipeline          |   Pure function: (state, actions) -> (state, obs, rewards, ...)
|  tick -> move -> interact   |
|  -> observe -> reward       |
+-----------------------------+
```

The **CoGridEnv** wrapper handles PettingZoo dict conversion and rendering.
The **autowire layer** discovers registered components and composes them into
configuration dicts. The **step pipeline** is a pure function that takes
an `EnvState` and actions, and returns the next state plus observations,
rewards, and done flags.

## The xp Backend System

CoGrid supports both NumPy and JAX through a shared array namespace called
`xp`. All simulation code uses `xp` operations, so the same code path works
on both backends with zero branching.

```python
from cogrid.backend import xp

# This works identically whether the backend is numpy or jax.numpy
arr = xp.zeros((3, 3), dtype=xp.int32)
result = xp.where(arr > 0, arr, xp.ones_like(arr))
```

Switch the backend before creating any environment:

=== "NumPy (default)"

    ```python
    # NumPy is the default -- no setup needed
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0")
    ```

=== "JAX"

    ```python
    from cogrid.backend import set_backend

    set_backend("jax")  # Must be called BEFORE creating environments
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0")
    ```

Under the hood, `xp` is a lazy proxy that resolves to `numpy` or `jax.numpy`
depending on which backend is active. The first environment created locks the
backend for the process -- subsequent environments must use the same backend.

!!! tip "Writing backend-compatible code"
    Use `xp.where(condition, a, b)` instead of Python `if/else` on array
    values. JAX traces through code at compile time and cannot handle
    Python-level conditionals on traced values. The `xp.where` pattern
    works identically on both backends.

## EnvState

All simulation state is held in a single frozen dataclass:
[`EnvState`][cogrid.backend.env_state.EnvState].

```python
from cogrid.backend.env_state import EnvState
```

### Core Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `agent_pos` | `(n_agents, 2)` int32 | Row, column position per agent |
| `agent_dir` | `(n_agents,)` int32 | Direction enum per agent |
| `agent_inv` | `(n_agents, 1)` int32 | Held item type ID (-1 = empty) |
| `wall_map` | `(H, W)` int32 | 1 where walls exist |
| `object_type_map` | `(H, W)` int32 | Object type ID at each cell |
| `object_state_map` | `(H, W)` int32 | Object state value at each cell |
| `extra_state` | `dict[str, array]` | Scope-prefixed environment-specific arrays |
| `rng_key` | `(2,)` uint32 | JAX PRNG key (None on NumPy) |
| `time` | `()` int32 | Current timestep |
| `done` | `(n_agents,)` bool | True for agents whose episode ended |

### Static Fields

These are compile-time constants (not traced through JAX JIT):

| Field | Description |
|-------|-------------|
| `n_agents` | Number of agents |
| `height` | Grid height |
| `width` | Grid width |
| `action_set` | `"cardinal"` or `"rotation"` |

### Extra State

The `extra_state` dict holds environment-specific arrays with scope-prefixed
keys. For example, the Overcooked environment stores pot cooking timers as
`"overcooked.pot_timer"`.

```python
import dataclasses
from cogrid.backend.env_state import get_extra, replace_extra

# Read extra state
timer = get_extra(state, "pot_timer", scope="overcooked")

# Update extra state (returns a new EnvState -- immutable)
state = replace_extra(state, "pot_timer", new_timer, scope="overcooked")

# Or use dataclasses.replace directly
new_extra = {**state.extra_state, "overcooked.pot_timer": new_timer}
state = dataclasses.replace(state, extra_state=new_extra)
```

### StateView

Reward functions and feature extractors receive a
[`StateView`][cogrid.backend.state_view.StateView] instead of the raw
`EnvState`. StateView provides the same dot access for core fields but
strips scope prefixes from extra state keys:

```python
# Inside a reward's compute() method:
# prev_state and state are StateView objects
timer = state.pot_timer  # instead of state.extra_state["overcooked.pot_timer"]
positions = state.agent_pos  # core fields work the same way
```

### JAX Pytree Registration

On the JAX backend, both `EnvState` and `StateView` are registered as JAX
pytrees. This means they can flow through `jax.jit`, `jax.vmap`, and
`jax.lax.scan` without any special handling.

!!! note "Immutability"
    `EnvState` is a frozen dataclass. You cannot mutate it directly. Use
    `dataclasses.replace()` to create a new state with updated fields.
    This is required for JAX compatibility and ensures clean state
    transitions in the step pipeline.

## Component API

CoGrid environments are assembled from three kinds of registered components:
**grid objects**, **rewards**, and **features**. Each uses a decorator to
register itself into a global registry, and the autowire layer discovers them
at initialization time.

### Registration Decorators

| Decorator | Registers | Module |
|-----------|-----------|--------|
| `@register_object_type` | Grid object classes | `cogrid.core.grid_object` |
| `@register_reward_type` | Reward subclasses | `cogrid.core.rewards` |
| `@register_feature_type` | Feature subclasses | `cogrid.core.features` |

Each decorator takes an `object_id` (or reward/feature name) and an optional
`scope` parameter that namespaces the component to avoid collisions between
environments.

### Grid Object Classmethods

Grid objects can declare behavior through classmethods that the autowire layer
discovers and wires into the step pipeline:

| Classmethod | Returns | Purpose |
|-------------|---------|---------|
| `build_tick_fn` | `(state, scope_config) -> state` | Per-step state updates (e.g., cooking timers) |
| `extra_state_schema` | `dict` | Declare extra state arrays this object needs |
| `extra_state_builder` | `fn(parsed, scope) -> dict` | Initialize extra state from the parsed layout |
| `build_static_tables` | `dict` | Additional lookup arrays for interaction logic |
| `build_render_sync_fn` | `fn(grid, state, scope) -> None` | Sync array state back to GridObj for rendering |

### How Autowire Works

At environment initialization, the autowire layer:

1. Scans the object registry for all objects in the environment's scope
2. Collects classmethods (`build_tick_fn`, `extra_state_schema`, etc.)
3. Builds a `scope_config` dict containing tick handlers, interaction tables,
   type ID mappings, and static lookup tables
4. Builds a `reward_config` from registered reward components
5. Builds a `feature_config` by composing registered feature functions

The resulting configs are closed over by `build_step_fn` and `build_reset_fn`,
producing pure functions that capture all environment logic. On the JAX
backend, these closures are JIT-compiled as a single unit.

For a hands-on walkthrough of creating components, see the
[Custom Environment Tutorial](../tutorials/custom-environment.md).

## Step Pipeline

The step pipeline is the heart of CoGrid's simulation engine. It is a pure
function that transforms state:

```
(EnvState, actions) -> (EnvState, obs, rewards, terminateds, truncateds, info)
```

### Pipeline Order

Each call to `step()` executes these stages in order:

```text
1. tick       -- Run scope-specific tick handler (e.g., decrement cooking timers)
2. move       -- Resolve agent movement with collision detection
3. interact   -- Process pickup/drop/place interactions
4. observe    -- Compose feature functions into observation arrays
5. reward     -- Compute rewards from all registered Reward components
6. done       -- Evaluate termination and truncation conditions
```

### Build Functions

The pipeline is constructed at init time through factory functions:

- **`build_step_fn`** closes over `scope_config`, `lookup_tables`,
  `feature_fn`, `reward_config`, and other static configuration. It returns
  a `(state, actions) -> ...` closure.

- **`build_reset_fn`** closes over `layout_arrays`, `spawn_positions`, and
  `feature_fn`. It returns a `(rng) -> (state, obs)` closure.

On the JAX backend, both closures are automatically JIT-compiled. The entire
step function -- from tick through reward computation -- compiles as one
XLA computation for maximum performance.

=== "NumPy"

    ```python
    # PettingZoo API (stateful wrapper calls pipeline internally)
    obs, rewards, terminateds, truncateds, info = env.step(actions)
    ```

=== "JAX (functional)"

    ```python
    # Direct functional API -- pure function on EnvState
    state, obs, rew, terms, truncs, info = env.jax_step(state, actions)
    ```

!!! tip "JIT compilation"
    On the JAX backend, the first call to `jax_step` or `jax_reset` triggers
    XLA compilation. Subsequent calls execute the compiled code and are
    significantly faster. See the [JAX Backend Tutorial](../tutorials/jax-backend.md)
    for details on JIT and vmap usage.
