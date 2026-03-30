<div style="text-align: center;">
  <img src="assets/images/cogrid_logo_clean.png" alt="CoGrid Logo" width="200">
</div>

# CoGrid

*A framework for multi-agent grid-world research*

CoGrid lets you build multi-agent grid-world environments for reinforcement learning.
It ships with dual NumPy and JAX backends, so the same environment runs interactively
on NumPy or at scale with JIT compilation and `vmap` batching. All environments
implement the [PettingZoo](https://pettingzoo.farama.org/) ParallelEnv API.

```bash
pip install cogrid
```

=== "NumPy"

    ```python
    from cogrid.envs import registry
    import cogrid.envs.overcooked  # register components

    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, info = env.reset(seed=0)

    for _ in range(100):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminateds, truncateds, info = env.step(actions)
    ```

=== "JAX"

    ```python
    import jax
    from cogrid.envs import registry
    import cogrid.envs.overcooked

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=0)
    n_agents = len(env.possible_agents)
    n_actions = len(env.action_set)

    def step_fn(carry, _):
        state, key = carry
        key, step_key, action_key = jax.random.split(key, 3)
        actions = {i: jax.random.randint(jax.random.fold_in(action_key, i), (), 0, n_actions)
                   for i in range(n_agents)}
        obs, state, rewards, terminated, truncated, info = env.jax_step(step_key, state, actions)
        return (state, key), rewards

    @jax.jit
    def rollout(key):
        key, reset_key = jax.random.split(key)
        obs, state, info = env.jax_reset(reset_key)
        (final_state, _), all_rewards = jax.lax.scan(
            step_fn, (state, key), None, length=env.max_steps,
        )
        return all_rewards  # {agent_id: (max_steps,)}

    rewards = rollout(jax.random.key(0))
    ```

## Architecture

Environments follow a component pipeline:

**Layout** -- ASCII grid parsed into state arrays.
**Objects** -- Registered types populate the grid and define interaction rules.
**Actions** -- Agents submit actions each step; the engine resolves movement and interactions.
**Step Pipeline** -- Updates state arrays, runs tick functions, computes rewards and observations.

All components are declared in a config dict and autowired by the engine. No manual wiring needed.

## Core Concepts

- **[Grid & Layouts](concepts/layouts.md)** -- 2D integer arrays, ASCII layout definitions, layout registry.
- **[Objects](concepts/objects.md)** -- `GridObj` base class, registration, containers, and recipes.
- **[Actions](concepts/actions.md)** -- Cardinal and rotation action sets, movement, pickup/drop/toggle pipeline.
- **[Observations](concepts/observations.md)** -- Composable feature extractors, per-agent vs global, built-in features.
- **[Rewards](concepts/rewards.md)** -- `Reward` and `InteractionReward` base classes, composition, coefficients.

## Environments

- **[Overcooked](environments/overcooked.md)** -- Cooperative cooking with 9 layout variants, recipes, and an optional order queue.
- **[Goal Seeking](environments/goal-seeking.md)** -- Navigation to valued goal cells.

## Next Steps

<div class="card-grid">
  <a href="getting-started/">
    <div class="card">
      <h3>Getting Started</h3>
      <p>Install CoGrid and run your first environment.</p>
    </div>
  </a>
  <a href="concepts/layouts/">
    <div class="card">
      <h3>Core Concepts</h3>
      <p>Grid layouts, objects, actions, observations, and rewards.</p>
    </div>
  </a>
  <a href="custom-environment/">
    <div class="card">
      <h3>Custom Environment</h3>
      <p>Build your own grid world from components.</p>
    </div>
  </a>
  <a href="jax-reference/">
    <div class="card">
      <h3>JAX Backend</h3>
      <p>JIT-compile and vmap-batch entire rollouts.</p>
    </div>
  </a>
</div>

## Citation

If you use CoGrid in your research, please cite the following paper:

```bibtex
@article{mcdonald2024cogrid,
  author  = {McDonald, Chase and Gonzalez, Cleotilde},
  title   = {CoGrid and Interactive Gym: A Framework for Multi-Agent Experimentation},
  year    = {forthcoming},
}
```

[Contributing Guide](https://github.com/chasemcd/cogrid/blob/main/CONTRIBUTING.md) -- How to contribute to CoGrid.
