<div style="text-align: center;">
  <img src="assets/images/cogrid_logo_clean.png" alt="CoGrid Logo" width="200">
</div>

# CoGrid

*A framework for multi-agent grid-world research with extremely high-speed simulation and a flexible backend for deployment.*

CoGrid lets you build multi-agent grid-world environments for reinforcement learning.
It ships with dual NumPy and JAX backends, so the same environment runs interactively
on NumPy or at scale with JIT compilation and `vmap` batching. All environments
implement the [PettingZoo](https://pettingzoo.farama.org/) ParallelEnv API, with a PettingZoo-like
functional API available for usage with JAX. 

### What CoGrid Offers

- **An Approachable API for Environment Customization** Building and customizing environments with JAX can be unintuitive and difficult, as the logic required for parallelization can be complex. Our goal was to abstract away a large poortion of this complexity to make it easier for researchers to customize their environments so that they can ask their own novel research questions. Users can define objects, interactions, and layouts through a declarative config. 
- **Compatibility with standard tooling.** All environments implement the [PettingZoo](https://pettingzoo.farama.org/) ParallelEnv API, so they work directly with training libraries, etc. A similar functional API is exposed for users who wish to take advantage of the full speedup that is offered by hardware-accelerated parallelization. 
- **Pre-built environments.** CoGrid ships with a suite of pre-built environments, inspired by prior work (importantly, they are **not** exact replicas!):
    - [Overcooked](environments/overcooked.md): the 5 original layouts from [Carroll et al., 2019](https://arxiv.org/abs/1910.05789) plus extended variants with multiple recipes and order queues.
    - [OvercookedV2](environments/overcooked_v2.md): 7 coordination benchmarks with asymmetric information and stochastic recipes from [Gessler et al., 2025](https://arxiv.org/abs/2503.17821).
    - [Goal Seeking](environments/goal-seeking.md): a simple debugging environment to navigate to targets. 

### Why a dual backend?

CoGrid implements and operates through a dual backend to avoid the rigidy of JAX execution and allow for the flexibility of a NumPy backend. This was largely done for one reason: we wanted to have environments that would train extremely fast (inspired by, e.g., JaxMARL), interact with them in standard Python libraries and frameworks, and *use them in online human-AI experiments*. This last point is critical: CoGrid was designed to satisfy the first two desiderata (which libraries like JaxMARL already do), but to also be used in the [Multi-User Gymnasium (MUG)](https://multi-user-gymnasium.readthedocs.io). 

MUG is a library that translates Gymnasium and PettingZoo environments to online multi-player experiments. For the best performance, these environments must be written in pure Python or have C-extensions that can have wheels build for WASM/Emscripten. MUG then runs the environments directly in the browser and coordinates multi-player experiments that can be used for research. More details around MUG can be found on [GitHub (MUG)](https://www.github.com/chasemcd/mug/) or on the corresponding [documentation site](https://multi-user-gymnasium.readthedocs.io).


## Installation

=== "Basic (NumPy)"

    ```bash
    pip install cogrid
    ```

=== "With JAX"

    ```bash
    pip install cogrid[jax]
    ```

For GPU support, see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quick Start

=== "NumPy"

    ```python
    from cogrid.envs import registry
    import cogrid.envs.overcooked  # register components

    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, info = env.reset(seed=0)

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
- **[Interactions](concepts/interactions.md)** -- Branch functions, `InteractionContext`, built-in and custom interactions.
- **[Actions](concepts/actions.md)** -- Cardinal and rotation action sets, movement, pickup/drop/toggle pipeline.
- **[Observations](concepts/observations.md)** -- Composable feature extractors, per-agent vs global, built-in features.
- **[Rewards](concepts/rewards.md)** -- `Reward` and `InteractionReward` base classes, composition, coefficients.

## Citation

If you use CoGrid in your research, please cite the following paper:

```bibtex
@article{mcdonald2026cogrid,
  title={CoGrid \& the Multi-User Gymnasium: A Framework for Multi-Agent Experimentation},
  author={McDonald, Chase and Gonzalez, Cleotilde},
  journal={arXiv preprint arXiv:2604.15044},
  year={2026}
}
```

[Contributing Guide](https://github.com/chasemcd/cogrid/blob/main/CONTRIBUTING.md) -- How to contribute to CoGrid.
