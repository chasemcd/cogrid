# JAX Backend

JIT-compile steps, batch environments with vmap, and train at GPU speed.

```python
import jax
import jax.numpy as jnp
import dataclasses
from cogrid.envs import registry
from cogrid.backend import xp
import cogrid.envs.overcooked  # register Overcooked components
```

**JIT compilation**

`jax_reset` and `jax_step` are automatically JIT-compiled on the JAX backend.
The first call triggers XLA compilation; subsequent calls execute the compiled
kernel and are orders of magnitude faster. Call `env.reset(seed=0)` once before
using `jax_reset`/`jax_step` -- this builds the compiled pipeline.

```python
env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
env.reset(seed=0)  # builds the JIT-compiled pipeline
n_agents = len(env.possible_agents)
n_actions = len(env.action_set)

key = jax.random.key(0)
obs, state, info = env.jax_reset(key)  # obs: (n_agents, obs_dim)

key, subkey = jax.random.split(key)
actions = jax.random.randint(subkey, (n_agents,), 0, n_actions)
obs, state, rewards, terminated, truncated, info = env.jax_step(state, actions)
# rewards: (n_agents,)
```

**vmap batching**

`jax.vmap` runs many environments in parallel with a single function call. Wrap `jax_reset` and `jax_step` in `jax.jit(jax.vmap(...))` to batch across thousands of environments on GPU. Each environment needs its own PRNG key -- use `jax.random.split` to create a batch from a single root key.

```python
n_envs = 1024
n_agents = len(env.possible_agents)
n_actions = len(env.action_set)

batched_reset = jax.jit(jax.vmap(env.jax_reset))
batched_step = jax.jit(jax.vmap(env.jax_step))

def step_fn(carry, _):
    states, key = carry
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_envs, n_agents), 0, n_actions)
    obs, states, rewards, terminated, truncated, info = batched_step(states, actions)
    return (states, key), rewards

@jax.jit
def batched_rollout(key):
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_envs)
    obs, states, info = batched_reset(reset_keys)
    (final_states, _), all_rewards = jax.lax.scan(
        step_fn, (states, key), None, length=env.max_steps,
    )
    return all_rewards  # (max_steps, n_envs, n_agents)

all_rewards = batched_rollout(jax.random.key(0))
print(f"Total reward across {n_envs} envs: {float(all_rewards.sum()):.1f}")
```

**Compatibility rules**

When writing tick functions, interactions, rewards, or features for the JAX backend, these rules apply. JAX traces your code at compile time, so anything that depends on runtime array values must use functional patterns.

Don't branch on array values with Python `if` -- the value isn't known at trace time. Use `xp.where` instead.

```python
# Don't
def tick(state, scope_config):
    if state.extra_state["my_env.timer"] > 0:  # fails under JIT
        return dataclasses.replace(state, time=state.time + 1)
    return state

# Do
def tick(state, scope_config):
    timer = state.extra_state["my_env.timer"]
    new_time = xp.where(timer > 0, state.time + 1, state.time)
    return dataclasses.replace(state, time=new_time)
```

Don't use side effects or dynamic shapes inside JIT-compiled functions. `print()` runs at trace time (not step time), and `xp.nonzero()` returns a dynamic-length array that XLA cannot handle.

```python
# Don't
def tick(state, scope_config):
    print(f"Step {state.time}")  # runs once at compile time, not every step
    idxs = xp.nonzero(state.object_type_map == 3)  # dynamic shape -- fails
    return state

# Do
def tick(state, scope_config):
    mask = state.object_type_map == 3  # static shape boolean mask
    count = xp.sum(mask)
    new_extra = {**state.extra_state, "my_env.target_count": count}
    return dataclasses.replace(state, extra_state=new_extra)
```

Don't mutate state directly -- `EnvState` is a frozen dataclass. Use `dataclasses.replace()` to create new state with updated fields, and return all changes through the state object.

```python
# Don't
def tick(state, scope_config):
    state.time = state.time + 1  # frozen dataclass -- raises error
    my_list.append(state.time)   # side effect lost after compilation
    return state

# Do
def tick(state, scope_config):
    new_time = state.time + 1
    return dataclasses.replace(state, time=new_time)
```

**Training integration**

CoGrid's pure functional API integrates directly with JAX transformations like `jit`, `vmap`, `lax.scan`, and `grad`. The step function is just a function -- compose it with any JAX primitive.

```python
env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
env.reset(seed=0)

step_fn = jax.jit(jax.vmap(env.jax_step))
reset_fn = jax.jit(jax.vmap(env.jax_reset))
```

For a complete training loop, see [`train_overcooked_jax.py`](https://github.com/chasemcd/cogrid/blob/main/examples/train_overcooked_jax.py).
