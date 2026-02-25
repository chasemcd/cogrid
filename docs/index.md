<div style="text-align: center;">
  <img src="assets/images/cogrid_logo_nobg.png" alt="CoGrid Logo" width="200">
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
    obs, info = env.reset(seed=42)

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, info = env.step(actions)
    ```

=== "JAX"

    ```python
    import jax
    from cogrid.envs import registry
    import cogrid.envs.overcooked

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    state, obs = env.jax_reset(jax.random.key(0))

    for _ in range(100):
        actions = jax.numpy.array([0, 3], dtype=jax.numpy.int32)
        state, obs, rewards, terms, truncs, info = env.jax_step(state, actions)
    ```

## Key Features

- **Component API** -- Build environments from [GridObject, Reward, and Feature](custom-environment.md) subclasses that the engine autowires into array-level code.
- **JAX Backend** -- [JIT-compile and vmap-batch](jax-reference.md) entire rollouts for high-throughput training.
- **PettingZoo ParallelEnv** -- Standard multi-agent interface compatible with [PettingZoo](https://pettingzoo.farama.org/) and existing RL libraries.
- **Config-Driven Environments** -- Define layouts, rewards, and features through a [config dict](overcooked.md) instead of writing boilerplate.

## Next Steps

<div class="card-grid">
  <a href="getting-started/">
    <div class="card">
      <h3>Getting Started</h3>
      <p>Install CoGrid and run your first environment.</p>
    </div>
  </a>
  <a href="custom-environment/">
    <div class="card">
      <h3>Custom Environment</h3>
      <p>Build your own grid world with objects, rewards, and features.</p>
    </div>
  </a>
  <a href="jax-reference/">
    <div class="card">
      <h3>Reference</h3>
      <p>JAX backend, advanced patterns, and the Overcooked environment.</p>
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
