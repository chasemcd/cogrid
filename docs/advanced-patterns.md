# Advanced Patterns

This page extends the goal-finding environment from [Custom Environment](custom-environment.md) with tick functions, custom interactions, extra state, and custom observations.

Shared imports for all examples below:

```python
import dataclasses
import functools

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at_2d
from cogrid.core.grid_object import GridObj, register_object_type, when
from cogrid.core.constants import Colors
from cogrid.core import layouts
from cogrid.core.rewards import InteractionReward
from cogrid.core.features import Feature, register_feature_type
from cogrid.core.interaction_context import clear_facing_cell, increment
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs import registry
```

## Tick functions

A `build_tick_fn` classmethod on a GridObj subclass returns a closure with signature `fn(state, scope_config) -> state`. It runs once per step, before movement and interactions. The autowire system discovers tick functions from all registered components automatically.

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

When an agent performs `PickupDrop` or `Toggle`, the pipeline runs each function in the `interactions` list. Each function receives an `InteractionContext` and returns `(should_apply, changes)`. The pipeline applies the changes from every function whose `should_apply` is true. If multiple functions fire for the same agent, the later function's changes overwrite earlier ones for overlapping keys.

`PickupDrop` ("pick up / put down") and `Toggle` ("activate / use") are semantically different actions. Built-in branches (pickup, drop, place) only fire on `PickupDrop`. Custom branches can check `ctx.action` against `ctx.action_id` to distinguish which action the agent chose.

### Signature

```python
def my_interaction(ctx):
    should_apply = ...  # bool: should this interaction happen?
    changes = {...}     # dict: what to change if it does
    return should_apply, changes
```

### InteractionContext

The pipeline builds an `InteractionContext` before calling interaction functions. Standard fields:

| Field | Type | Meaning |
|-------|------|---------|
| `can_interact` | bool | `True` if this agent performed `PickupDrop` or `Toggle` and no other agent blocks the cell ahead. |
| `action` | int | Raw action index this agent chose this step. |
| `action_id` | `ActionID` | Named indices for all actions in this env. Use `ctx.action_id.pickup_drop`, `ctx.action_id.toggle`, etc. Actions not in the action set have index `-1`. |
| `facing_row` | int | Row of the cell the agent is facing. |
| `facing_col` | int | Column of the cell the agent is facing. |
| `facing_type` | int | Type ID of the object in the faced cell (0 = empty). |
| `agent_index` | int | Which agent (0, 1, ...) is acting. |
| `held_item` | int | Type ID of item the agent holds. `-1` = empty hands. |
| `type_ids` | dict | Maps object names to integer type IDs. `ctx.type_ids["goal"]` returns the goal's type ID. |
| `object_type_map` | `(H, W)` int array | Grid of object type IDs. |
| `object_state_map` | `(H, W)` int array | Grid of per-cell state values. |
| `agent_inv` | `(n_agents, 1)` int array | All agents' inventories. |

Extra-state arrays declared by components (via `extra_state_schema`) are available directly as attributes on `ctx`. For example, if a component declares `goals_collected`, access it as `ctx.goals_collected`.

### Helper functions

Import from `cogrid.core.interaction_context`:

| Helper | Returns | Purpose |
|--------|---------|---------|
| `clear_facing_cell(ctx)` | `object_type_map` | Set the faced cell to empty (type 0). |
| `set_facing_cell(ctx, type_id)` | `object_type_map` | Set the faced cell to a specific type. |
| `pickup_from_facing_cell(ctx)` | `(object_type_map, agent_inv)` | Pick up the faced object into the agent's inventory. |
| `place_in_facing_cell(ctx)` | `(object_type_map, agent_inv)` | Place the held item in the faced cell. |
| `give_item(ctx, type_id)` | `agent_inv` | Put an item in the agent's inventory. |
| `empty_hands(ctx)` | `agent_inv` | Clear the agent's inventory. |
| `increment(array, index)` | array | `array[index] += 1`. Works on both numpy and JAX. |
| `find_facing_instance(positions, row, col)` | `(index, is_match)` | Find which instance of a multi-position object the agent faces. |

### Example: collecting a goal (PickupDrop)

```python
from cogrid.core.interaction_context import clear_facing_cell, increment


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
