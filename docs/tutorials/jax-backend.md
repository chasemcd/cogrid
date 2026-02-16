# JAX Backend

This tutorial covers CoGrid's JAX backend: the functional API, JIT compilation,
and vmap batching for high-performance training. By the end you will be able to
run thousands of environments in parallel on a single GPU.

## Prerequisites

Install CoGrid with JAX support:

```bash
pip install cogrid[jax]
```

Switch to the JAX backend **before** creating any environment:

```python
from cogrid.backend import set_backend

set_backend("jax")
```

!!! warning "Backend must be set first"
    `set_backend("jax")` must be called before any call to `registry.make()`.
    The first environment created locks the backend for the entire process.
    Attempting to mix backends raises a `RuntimeError`.

## Functional API

CoGrid provides two APIs for interacting with environments:

1. **PettingZoo API** (stateful wrapper) -- identical to NumPy usage
2. **Functional API** -- pure functions on `EnvState`, designed for JIT/vmap

The functional API exposes two methods on the environment object:

| Method | Signature | Description |
|--------|-----------|-------------|
| `env.jax_reset` | `(rng_key) -> (EnvState, obs)` | Initialize state from a JAX PRNG key |
| `env.jax_step` | `(EnvState, actions) -> (EnvState, obs, rew, terms, truncs, info)` | Advance one timestep |

Both are pure functions: they take state in, return new state out, with no
side effects. This makes them compatible with JAX's transformation system.

### Basic Usage

```python
import jax
import jax.numpy as jnp
from cogrid.backend import set_backend
from cogrid.envs import registry
import cogrid.envs.overcooked  # register components

set_backend("jax")
env = registry.make("Overcooked-CrampedRoom-V0")

# Must call reset() once to build the pipeline
obs, info = env.reset(seed=42)

# Access the functional API
reset_fn = env.jax_reset
step_fn = env.jax_step

# Reset with a JAX PRNG key
key = jax.random.key(0)
state, obs = reset_fn(key)

# Step with an action array (one int per agent)
actions = jnp.array([0, 3], dtype=jnp.int32)  # Agent 0: Up, Agent 1: Right
state, obs, rew, terminateds, truncateds, info = step_fn(state, actions)
```

!!! note "PettingZoo API still works"
    You can use the standard `env.step(actions_dict)` and `env.reset()` on
    the JAX backend too. The stateful wrapper calls the functional pipeline
    internally and converts between dict and array formats. The functional
    API is for when you need direct control over state for JIT/vmap.

### Contrast with NumPy

=== "NumPy (stateful)"

    ```python
    # Dict-based, stateful
    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, info = env.reset(seed=42)

    actions = {0: 0, 1: 3}
    obs, rewards, terminateds, truncateds, info = env.step(actions)
    ```

=== "JAX (functional)"

    ```python
    # Array-based, pure functional
    state, obs = env.jax_reset(jax.random.key(0))

    actions = jnp.array([0, 3], dtype=jnp.int32)
    state, obs, rew, terms, truncs, info = env.jax_step(state, actions)
    ```

The functional API uses arrays instead of dicts: actions are `(n_agents,)`
int32 arrays, observations are `(n_agents, obs_dim)` arrays, and rewards are
`(n_agents,)` float32 arrays.

## JIT Compilation

The functional step and reset functions are **automatically JIT-compiled** when
the JAX backend is active. The first call triggers XLA compilation; subsequent
calls execute the compiled code.

```python
import time

state, obs = reset_fn(jax.random.key(0))
actions = jnp.array([0, 3], dtype=jnp.int32)

# First call: includes compilation time
t0 = time.time()
state, obs, rew, terms, truncs, info = step_fn(state, actions)
print(f"First call (compile + execute): {time.time() - t0:.3f}s")

# Subsequent calls: fast
t0 = time.time()
for _ in range(1000):
    state, obs, rew, terms, truncs, info = step_fn(state, actions)
print(f"1000 steps: {time.time() - t0:.3f}s")
```

=== "Without JIT"

    ```python
    # If you need to disable auto-JIT (debugging):
    from cogrid.core.step_pipeline import build_step_fn

    step_fn = build_step_fn(..., jit_compile=False)
    ```

=== "With JIT (default)"

    ```python
    # Auto-JIT is the default on JAX backend.
    # env.jax_step is already JIT-compiled.
    step_fn = env.jax_step
    ```

!!! tip "Warmup in benchmarks"
    Always call the function once before timing to exclude compilation.
    JIT compilation happens once per unique combination of static arguments
    (like `n_agents`, `height`, `width`). Recompilation only occurs if you
    create an environment with different static dimensions.

## vmap Batching

`jax.vmap` lets you run many environments in parallel with a single function
call. This is the key to high-throughput training -- thousands of environments
execute simultaneously on GPU/TPU.

### Basic Pattern

```python
import jax
import jax.numpy as jnp
from cogrid.backend import set_backend
from cogrid.envs import registry
import cogrid.envs.overcooked

set_backend("jax")
env = registry.make("Overcooked-CrampedRoom-V0")
obs, info = env.reset(seed=42)

n_envs = 1024

# Split one key into n_envs independent keys
keys = jax.random.split(jax.random.key(0), n_envs)

# Batch reset: returns (n_envs, ...) arrays
batched_reset = jax.jit(jax.vmap(env.jax_reset))
states, obs = batched_reset(keys)

# Batch step: actions shape is (n_envs, n_agents)
batched_step = jax.jit(jax.vmap(env.jax_step))
actions = jnp.zeros((n_envs, 2), dtype=jnp.int32)
states, obs, rew, terms, truncs, info = batched_step(states, actions)
```

### Rollout Loop

Run multiple steps across all batched environments:

```python
n_steps = 50
action_key = jax.random.key(42)
total_reward = jnp.float32(0.0)

for step_i in range(n_steps):
    action_key, subkey = jax.random.split(action_key)
    actions = jax.random.randint(subkey, (n_envs, 2), 0, 4)
    states, obs, rew, terms, truncs, info = batched_step(states, actions)
    total_reward += rew.sum()

avg_reward = total_reward / n_envs
print(f"Average reward over {n_steps} steps: {float(avg_reward):.1f}")
```

!!! tip "Key splitting"
    Each environment needs an independent PRNG key for reproducible
    randomization. Use `jax.random.split(key, n_envs)` to create a batch
    of keys from a single root key.

### Full Example

The `examples/goal_finding.py` script demonstrates the complete pattern:
single environment on NumPy, single environment on JAX, and vmap over 1024
environments. Run it with:

```bash
python examples/goal_finding.py
```

## Training Example

The `examples/train_overcooked_jax.py` script demonstrates a complete
shared-parameter IPPO training loop on Overcooked using CoGrid's JAX backend.
Key features:

- `jax.vmap` for parallel environment rollouts
- `jax.lax.scan` for the training loop (no Python for-loop overhead)
- Auto-reset on episode completion
- The entire training loop compiles as a single XLA computation

The script is based on [JaxMARL](https://github.com/FLAIROx/JaxMARL) and
adapted for CoGrid. Run it with:

```bash
python examples/train_overcooked_jax.py
```

The key takeaway is that CoGrid's pure functional step pipeline integrates
naturally with JAX's transformation primitives. The step function is just
a function -- it works with `jit`, `vmap`, `scan`, `grad`, and any other
JAX transformation.

## JAX Compatibility Rules

When writing components (tick functions, interaction functions, rewards,
features) that will run on the JAX backend, follow these rules:

!!! warning "No Python control flow on traced values"
    JAX traces through your code at compile time. Python `if/else`
    statements on array values will fail because the array value is not
    known at trace time.

    ```python
    # WRONG -- fails under JIT
    if state.agent_inv[0, 0] > 0:
        result = do_something()

    # RIGHT -- works on both backends
    result = xp.where(state.agent_inv[0, 0] > 0, do_something(), default)
    ```

!!! warning "Static array shapes"
    All array shapes must be known at compile time. You cannot create arrays
    whose size depends on runtime values.

    ```python
    # WRONG -- dynamic shape
    mask = xp.nonzero(state.object_type_map == target_id)

    # RIGHT -- static shape, boolean indexing
    mask = (state.object_type_map == target_id)
    count = xp.sum(mask)
    ```

!!! warning "No side effects in JIT"
    JIT-compiled code cannot perform I/O, modify global variables, or have
    any Python-level side effects. All state changes must go through the
    returned `EnvState`.

    ```python
    # WRONG -- side effect
    def tick(state, scope_config):
        print(f"Step {state.time}")  # This runs at compile time, not runtime
        ...

    # RIGHT -- pure function
    def tick(state, scope_config):
        timer = state.extra_state["my_env.timer"]
        new_timer = xp.where(timer > 0, timer - 1, timer)
        new_extra = {**state.extra_state, "my_env.timer": new_timer}
        return dataclasses.replace(state, extra_state=new_extra)
    ```

!!! warning "Use dataclasses.replace for state updates"
    `EnvState` is a frozen dataclass. Use `dataclasses.replace()` to create
    a new state with updated fields. Direct attribute assignment is forbidden.

    ```python
    import dataclasses

    # WRONG -- mutation
    state.time = state.time + 1

    # RIGHT -- immutable update
    state = dataclasses.replace(state, time=state.time + 1)
    ```

**Summary of rules:**

| Rule | Pattern |
|------|---------|
| No `if/else` on arrays | Use `xp.where(cond, a, b)` |
| Static shapes | All shapes known at compile time |
| No side effects | No print, no globals, no I/O in JIT |
| Immutable state | Use `dataclasses.replace()` |
| PRNG keys | Split keys explicitly, never reuse |
