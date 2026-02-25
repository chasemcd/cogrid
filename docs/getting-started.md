# Getting Started

Install CoGrid and run a multi-agent Overcooked environment in under a minute.

=== "Basic (NumPy)"

    ```bash
    pip install cogrid
    ```

=== "With JAX"

    ```bash
    pip install cogrid[jax]
    ```

For GPU support, see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

=== "NumPy"

    ```python
    from cogrid.envs import registry
    import cogrid.envs.overcooked

    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, info = env.reset(seed=42)  # obs: dict[str, ndarray]

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminateds, truncateds, info = env.step(actions)
        # obs: dict[str, ndarray], rewards: dict[str, float]
    ```

=== "JAX"

    ```python
    import jax
    from cogrid.envs import registry
    import cogrid.envs.overcooked

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=0)  # builds JIT-compiled functions

    state, obs = env.jax_reset(jax.random.key(0))  # obs: ndarray (n_agents, obs_dim)
    actions = jax.numpy.array([0, 3], dtype=jax.numpy.int32)
    state, obs, rewards, terminateds, truncateds, info = env.jax_step(state, actions)
    # rewards: ndarray (n_agents,)
    ```
