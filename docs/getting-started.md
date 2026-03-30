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
    ```

=== "JAX"

    ```python
    import jax
    from cogrid.envs import registry
    import cogrid.envs.overcooked

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=0)  # builds JIT-compiled functions
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

!!! tip ""
    **Next: [Custom Environment](custom-environment.md)**
