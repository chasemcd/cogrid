# Getting Started

This guide walks through installing CoGrid, creating an environment, and running
your first simulation loop on both the NumPy and JAX backends.

## Installation

=== "Basic (NumPy)"

    ```bash
    pip install cogrid
    ```

=== "With JAX"

    ```bash
    pip install cogrid[jax]
    ```

=== "Development"

    ```bash
    git clone https://github.com/chasemcd/cogrid.git
    cd cogrid
    pip install -e ".[dev]"
    ```

## Quick Start

CoGrid environments follow the [PettingZoo](https://pettingzoo.farama.org/)
parallel API. Create an environment, reset it, and step through a loop:

=== "NumPy (default)"

    ```python
    from cogrid.envs import registry
    import cogrid.envs.overcooked  # triggers component registration

    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, info = env.reset(seed=42)

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminateds, truncateds, info = env.step(actions)
    ```

=== "JAX"

    ```python
    import jax
    from cogrid.backend import set_backend
    from cogrid.envs import registry
    import cogrid.envs.overcooked

    set_backend("jax")
    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, info = env.reset(seed=42)

    # Use the functional API for JIT/vmap
    state, obs_arr = env.jax_reset(jax.random.key(0))
    actions = jax.numpy.array([0, 3], dtype=jax.numpy.int32)
    state, obs_arr, rew, terms, truncs, info = env.jax_step(state, actions)
    ```

!!! note "Import side effects"
    You must `import cogrid.envs.overcooked` (or whichever environment module
    you want) **before** calling `registry.make()`. The import triggers
    `@register_object_type` and `@register_feature_type` decorators that
    register grid objects and features into CoGrid's engine. Reward subclasses
    do not require decorator registration -- they are instantiated and passed
    directly in the config `"rewards"` list.

## Understanding the Output

After calling `env.step(actions)`, you get five return values:

| Value | Type | Description |
|-------|------|-------------|
| `obs` | `dict[AgentID, ndarray]` | Per-agent observation arrays |
| `rewards` | `dict[AgentID, float]` | Per-agent scalar rewards |
| `terminateds` | `dict[AgentID, bool]` | True if the agent's episode ended by task completion |
| `truncateds` | `dict[AgentID, bool]` | True if the agent's episode ended by time limit |
| `info` | `dict[AgentID, dict]` | Per-agent auxiliary information |

**Observations** are flat arrays composed from registered
[Feature](tutorials/custom-environment.md#step-7-define-feature-extractors)
functions. Each feature contributes a segment of the array, concatenated in
the order listed in the environment config. For example, the Overcooked
environments compose `agent_dir`, `agent_position`, and inventory features
into a single flat vector per agent.

**Rewards** are summed across all registered
[Reward](tutorials/custom-environment.md#step-6-define-reward-functions)
components. Each `Reward.compute()` returns an `(n_agents,)` float32 array;
the engine sums them and converts to per-agent floats.

## What's Next

- **[Architecture](concepts/architecture.md)** -- Understand the xp backend
  system, EnvState, component API, and step pipeline.
- **[Custom Environment Tutorial](tutorials/custom-environment.md)** -- Create
  your own environment with grid objects, rewards, and features.
- **[JAX Backend Tutorial](tutorials/jax-backend.md)** -- Use JIT compilation
  and vmap batching for high-performance training.
- **[API Reference](reference/cogrid/index.md)** -- Full auto-generated API documentation.
