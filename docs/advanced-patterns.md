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
from cogrid.core.interactions import (
    branch_pickup,
    branch_drop_on_empty,
    merge_branch_results,
)
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

When an agent performs the interact action (PickupDrop), the pipeline calls the `interaction_fn` for that agent. The function receives per-agent context and returns an updated state. Inside, it evaluates a sequence of **branch functions** — small pure functions that each handle one type of interaction (pick up an object, drop an item, collect a goal, etc.).

### The handled cascade

Branch functions are evaluated in order. Each has the signature:

```
(handled, ctx) -> (cond, updates, new_handled)
```

| Argument | Type | Meaning |
|----------|------|---------|
| `handled` | bool scalar | `True` if a previous branch already claimed this interact action. |
| `ctx` | dict | State arrays and static lookup tables the branch reads from. |
| **Return** | | |
| `cond` | bool scalar | Whether this branch fires. Always includes `~handled` so it won't fire if a prior branch already did. |
| `updates` | dict | Arrays to write back if `cond` is true (e.g., modified `object_type_map`). |
| `new_handled` | bool scalar | Updated flag: `handled \| cond`. Passed to the next branch. |

The first branch whose `cond` is true wins. Later branches see `handled=True` and skip. This ensures each agent does at most one thing per interact action.

### The ctx dict

The pipeline builds a `ctx` dict before running branches. Standard keys:

| Key | Value | Source |
|-----|-------|--------|
| `base_ok` | bool scalar | `True` if this agent performed the interact action **and** no other agent is blocking the cell ahead. Computed by `process_interactions` before calling `interaction_fn`. |
| `fwd_r`, `fwd_c` | int scalars | Row and column of the cell the agent is facing. |
| `fwd_type` | int scalar | `object_type_map[fwd_r, fwd_c]` — the type ID of the object in that cell. |
| `agent_idx` | int scalar | Which agent (0, 1, ...) is acting. |
| `inv_item` | int scalar | Type ID the agent is holding, or `-1` for empty hands. |
| `agent_inv` | `(n_agents, 1)` int array | Full inventory array (branches write to it via `set_at`). |
| `object_type_map` | `(H, W)` int array | Grid of object type IDs. |
| `object_state_map` | `(H, W)` int array | Grid of per-cell state values. |
| `CAN_PICKUP`, `PICKUP_FROM_GUARD`, `PLACE_ON_GUARD`, `pickup_from_produces` | int arrays | Static lookup tables built at init time. Control which objects can be picked up, dropped, or placed on surfaces. |

Custom branches can add extra keys (like `goal_id` and `goals_collected` below).

### Example branch: collecting a goal

```python
def branch_collect_goal(handled, ctx):
    """Remove a goal from the grid when an agent faces it."""
    # Fire only if:
    #  - no prior branch handled this action (~handled)
    #  - the interact action is valid for this agent (base_ok)
    #  - the cell ahead contains a goal
    cond = ~handled & ctx["base_ok"] & (ctx["fwd_type"] == ctx["goal_id"])

    # Clear the goal cell by setting its type_id to 0 (empty)
    otm = set_at_2d(ctx["object_type_map"], ctx["fwd_r"], ctx["fwd_c"], 0)

    # Increment this agent's collection counter.
    # .at[].add() is JAX syntax; the hasattr guard keeps numpy compat.
    gc = ctx["goals_collected"]
    gc = gc.at[ctx["agent_idx"]].add(1) if hasattr(gc, "at") else gc

    return cond, {"object_type_map": otm, "goals_collected": gc}, handled | cond
```

### Wiring branches into an interaction function

The `interaction_fn` builds the `ctx` dict from state arrays and static tables, runs each branch in priority order, and merges results. Custom branches go before generic ones — order determines priority.

```python
def build_goal_interaction_fn(scope):
    from cogrid.core.grid_object import object_to_idx

    # Look up the integer type_id for "goal" at init time (closed over).
    goal_id = object_to_idx("goal", scope=scope)

    def interaction_fn(state, agent_idx, fwd_r, fwd_c, base_ok, scope_config):
        st = scope_config.get("static_tables", {})

        # Read the goals_collected extra-state array
        gc_key = f"{scope}.goals_collected"
        gc = state.extra_state.get(gc_key, xp.zeros(2, dtype=xp.int32))

        # Build the context dict that branches read from.
        # Every branch expects the standard keys; custom branches
        # (branch_collect_goal) also need goal_id and goals_collected.
        ctx = {
            # Per-call info (set by process_interactions before this call)
            "base_ok": base_ok,
            "fwd_r": fwd_r,
            "fwd_c": fwd_c,
            "agent_idx": agent_idx,

            # Derived from state arrays
            "fwd_type": state.object_type_map[fwd_r, fwd_c],
            "inv_item": state.agent_inv[agent_idx, 0],

            # Goal-specific keys (custom to this environment)
            "goal_id": goal_id,
            "goals_collected": gc,

            # State arrays (branches may produce updated versions of these)
            "agent_inv": state.agent_inv,
            "object_type_map": state.object_type_map,
            "object_state_map": state.object_state_map,

            # Static lookup tables (built at init, never modified)
            "CAN_PICKUP": st["CAN_PICKUP"],
            "PICKUP_FROM_GUARD": st["PICKUP_FROM_GUARD"],
            "PLACE_ON_GUARD": st["PLACE_ON_GUARD"],
            "pickup_from_produces": st["pickup_from_produces"],
        }

        # Evaluate branches in priority order.
        # branch_collect_goal runs first — if it fires, the generic
        # branch_pickup and branch_drop_on_empty won't.
        handled = xp.bool_(False)
        results = []
        for fn in [branch_collect_goal, branch_pickup, branch_drop_on_empty]:
            c, u, handled = fn(handled, ctx)
            results.append((c, u))

        # merge_branch_results applies the winning branch's array updates.
        # For each key, it does: final = xp.where(cond, updated, original).
        merged = merge_branch_results(
            results,
            {
                "agent_inv": state.agent_inv,
                "object_type_map": state.object_type_map,
                "object_state_map": state.object_state_map,
            },
        )

        # goals_collected lives in extra_state, so merge it separately.
        extra = dict(state.extra_state)
        for c, u in results:
            if "goals_collected" in u:
                extra[gc_key] = xp.where(c, u["goals_collected"], gc)

        return dataclasses.replace(state, **merged, extra_state=extra)

    return interaction_fn
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
    "interaction_fn": build_goal_interaction_fn("global"),
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
