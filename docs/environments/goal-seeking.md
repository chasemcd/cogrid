# Goal Seeking

## Overview

Goal Seeking is a simple navigation environment where agents move through a
grid to reach goal positions. It serves as a minimal example of how to build a
CoGrid environment from scratch using the component API -- no environment
subclass is needed.

<!-- No screenshot available yet. Generate one with env.get_frame() or EnvRenderer. -->

## Environment Details

### Game Mechanics

- Agents spawn on designated start cells and navigate using cardinal actions
  (up, down, left, right).
- A **goal** cell (`g`) is placed on the grid. When any agent steps onto it,
  all agents receive +1.0 reward (common reward).
- The episode terminates when an agent reaches the goal or `max_steps` is
  reached.

### Objects

| Object | Char | Color | Description |
|--------|------|-------|-------------|
| Goal | `g` | Green | Target cell that agents navigate toward |
| Wall | `#` | Grey | Impassable boundary |

### Reward

The `GoalReward` component awards +1.0 (common) for each agent standing on a
goal cell at every step, encouraging agents to reach the goal quickly.

## Quick Start

=== "NumPy (default)"

    ```python
    from cogrid.envs import registry

    # The GoalFinding example registers "GoalFinding-Simple-V0"
    import examples.goal_finding  # registers objects, layout, reward, and env

    env = registry.make("GoalFinding-Simple-V0", backend="numpy")
    obs, info = env.reset(seed=42)

    for _ in range(50):
        actions = {a: env.action_space.sample() for a in env.possible_agents}
        obs, rewards, terminateds, truncateds, info = env.step(actions)
    ```

=== "JAX"

    ```python
    import jax
    import jax.numpy as jnp
    from cogrid.envs import registry
    import examples.goal_finding

    env = registry.make("GoalFinding-Simple-V0", backend="jax")

    # Direct functional API (JIT-compiled)
    step_fn = env.jax_step
    reset_fn = env.jax_reset

    state, obs = reset_fn(jax.random.key(0))
    actions = jnp.array([0, 3], dtype=jnp.int32)  # Agent 0: Up, Agent 1: Right
    state, obs, rew, terminateds, truncateds, _ = step_fn(state, actions)

    # Batched rollouts with vmap (1024 parallel environments)
    batched_reset = jax.jit(jax.vmap(reset_fn))
    batched_step = jax.jit(jax.vmap(step_fn))

    keys = jax.random.split(jax.random.key(1), 1024)
    batched_state, batched_obs = batched_reset(keys)
    ```

## Links

- [JAX Backend Tutorial](../tutorials/jax-backend.md) -- vmap, JIT, and the
  functional API in detail
- [Goal Finding Example](https://github.com/chasemcd/cogrid/blob/main/examples/goal_finding.py) --
  full source with NumPy and JAX demos
