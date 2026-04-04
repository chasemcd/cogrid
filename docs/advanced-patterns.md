# Advanced Patterns

This page extends the goal-finding environment from [Custom Environment](custom-environment.md) with tick functions, custom interactions, extra state, and custom observations.

Shared imports for all examples below:

```python
import dataclasses
import functools

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at_2d
from cogrid.core.objects import GridObj, register_object_type, when
from cogrid.core.constants import Colors
from cogrid.core.grid import layouts
from cogrid.core.pipeline.rewards import InteractionReward
from cogrid.core.features import Feature, register_feature_type
from cogrid.core.pipeline.context import clear_facing_cell, increment
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs import registry
```

## Tick functions

See [Tick Functions](concepts/objects.md#tick-functions) for the full concept reference. Below is the Goal object from this example, which uses a tick function to reposition itself every 10 steps:

```python
@register_object_type("goal")
class Goal(GridObj):
    color = Colors.Green
    char = "g"
    can_overlap = when()

    def __init__(self, **kwargs):
        super().__init__(state=0)

    @classmethod
    def build_tick_fn(cls):
        """Move the goal to a new cell every 10 steps."""

        def tick(state, scope_config):
            # Only trigger on multiples of 10, skip step 0
            cycle = state.time % 10 == 0
            H, W = state.wall_map.shape

            # Deterministic "random" position derived from the step counter
            new_r = (state.time * 3 + 1) % H
            new_c = (state.time * 7 + 3) % W

            # Only move if the target cell is free (not a wall)
            free = (state.wall_map[new_r, new_c] == 0) & cycle & (state.time > 0)

            # Place the goal's type_id at the new position.
            # scope_config["type_ids"] maps object names to their int IDs.
            goal_id = scope_config["type_ids"]["goal"]
            otm = xp.where(
                free,
                set_at_2d(state.object_type_map, new_r, new_c, goal_id),
                state.object_type_map,
            )
            return dataclasses.replace(state, object_type_map=otm)

        return tick

    @classmethod
    def extra_state_schema(cls):
        """Declare a per-agent counter for collected goals."""
        return {"goals_collected": {"shape": ("n_agents",), "dtype": "int32"}}

    @classmethod
    def extra_state_builder(cls):
        """Initialize goals_collected to zeros at reset time."""

        def builder(parsed_arrays, scope):
            import numpy as np
            return {f"{scope}.goals_collected": np.zeros(2, dtype=np.int32)}

        return builder
```

The pipeline calls all tick functions before movement and interactions each step.

## Interaction functions

See [Interactions](concepts/interactions.md) for the full concept reference — branch signature, `InteractionContext` fields, built-in branches, and helper functions.

Below are two custom interaction examples from this environment.

### Example: collecting a goal (PickupDrop)

```python
from cogrid.core.pipeline.context import clear_facing_cell, increment


def collect_goal(ctx):
    """Remove a goal when the agent picks it up."""
    goal_id = ctx.type_ids["goal"]
    is_pickup = ctx.action == ctx.action_id.pickup_drop
    should_apply = ctx.can_interact & is_pickup & (ctx.facing_type == goal_id)
    changes = {
        "object_type_map": clear_facing_cell(ctx),
        "goals_collected": increment(ctx.goals_collected, ctx.agent_index),
    }
    return should_apply, changes
```

### Example: toggling a door (Toggle)

```python
from cogrid.backend import xp
from cogrid.backend.array_ops import set_at_2d


def toggle_door(ctx):
    """Open or close a door when the agent toggles it."""
    door_id = ctx.type_ids["door"]
    is_toggle = ctx.action == ctx.action_id.toggle
    should_apply = ctx.can_interact & is_toggle & (ctx.facing_type == door_id)
    cur = ctx.object_state_map[ctx.facing_row, ctx.facing_col]
    new_state = xp.where(cur == 0, 1, 0)
    new_osm = set_at_2d(ctx.object_state_map, ctx.facing_row, ctx.facing_col, new_state)
    return should_apply, {"object_state_map": new_osm}
```

### Wiring into config

Pass a list of interaction functions in the config. User interactions run before any auto-discovered ones (from the autowire system).

```python
goal_config = {
    ...
    "interactions": [collect_goal],
}
```

## Extra state

`EnvState.extra_state` is a dict of scope-prefixed arrays for domain-specific data. Components declare what they need with two classmethods (shown on `Goal` above):

- `extra_state_schema()` — declares shape and dtype.
- `extra_state_builder()` — returns a factory that initializes the arrays at reset time.

Tick and interaction functions read and write these arrays at step time. Keys are prefixed with the scope name (e.g., `global.goals_collected`). Feature functions access them without the prefix via `StateView` — `state.goals_collected`.

## Feature extractors

Subclass `Feature`, decorate with `@register_feature_type`, set `per_agent` and `obs_dim`, and implement `build_feature_fn(cls, scope)`. The returned closure has signature `fn(state, agent_idx) -> ndarray`. The autowire system composes all registered features into a single observation vector.

```python
@register_feature_type("goals_collected", scope="global")
class GoalsCollected(Feature):
    per_agent = True
    obs_dim = 1

    @classmethod
    def build_feature_fn(cls, scope):
        def fn(state, agent_idx):
            # state.goals_collected is available because StateView
            # strips the scope prefix from extra_state keys.
            return xp.array([state.goals_collected[agent_idx]], dtype=xp.float32)
        return fn
```

Add `"goals_collected"` to the features list in the config. Features listed in the config are composed into a single `(obs_dim,)` vector per agent at init time.

## Putting it together

Register the layout, reward, termination function, and environment:

```python
class GoalReward(InteractionReward):
    action = None
    overlaps = "goal"


layouts.register_layout(
    "goal_v0",
    [
        "#######",
        "#  g  #",
        "# # # #",
        "#     #",
        "# # # #",
        "#+ + +#",
        "#######",
    ],
)


def goal_terminated(prev_state, state, reward_config):
    """Terminate when any agent stands on the goal cell."""
    goal_id = reward_config["type_ids"].get("goal", -1)
    rows = state.agent_pos[:, 0]
    cols = state.agent_pos[:, 1]
    return state.object_type_map[rows, cols] == goal_id


goal_config = {
    "name": "goal_finding",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": [
        "agent_dir",
        "agent_position",
        "can_move_direction",
        "inventory",
        "goals_collected",
    ],
    "rewards": [GoalReward(coefficient=1.0, common_reward=True)],
    "grid": {"layout": "goal_v0"},
    "max_steps": 50,
    "scope": "global",
    "terminated_fn": goal_terminated,
    "interactions": [collect_goal],
}

registry.register(
    "GoalFinding-V1",
    functools.partial(CoGridEnv, config=goal_config),
)
```

=== "NumPy"

    ```python
    env = registry.make("GoalFinding-V1")
    obs, info = env.reset(seed=42)

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminateds, truncateds, info = env.step(actions)
    ```

=== "JAX"

    ```python
    import jax

    env = registry.make("GoalFinding-V1", backend="jax")
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
        return all_rewards

    rewards = rollout(jax.random.key(0))
    ```

For a full-featured example of these patterns, see the [Overcooked environment source](https://github.com/chasemcd/cogrid/tree/main/cogrid/envs/overcooked).
