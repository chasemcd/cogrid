# Advanced Patterns

Add behaviors to your grid world.

This page extends the goal-finding environment from [Custom Environment](custom-environment.md) with tick functions, custom interactions, extra state, and custom observations.

```python
import dataclasses, functools
from cogrid.backend import xp
from cogrid.backend.array_ops import set_at_2d
from cogrid.core.grid_object import GridObj, register_object_type, when
from cogrid.core.constants import Colors
from cogrid.core import layouts
from cogrid.core.rewards import InteractionReward
from cogrid.core.features import Feature, register_feature_type
from cogrid.core.interactions import branch_pickup, branch_drop_on_empty, merge_branch_results
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs import registry
```

**Tick functions**

A `build_tick_fn` classmethod on a GridObj subclass returns a `fn(state, scope_config) -> state` closure that runs once per step before movement and interactions. The autowire system auto-discovers tick functions from all registered components.

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
            cycle = state.time % 10 == 0
            H, W = state.wall_map.shape
            new_r, new_c = (state.time * 3 + 1) % H, (state.time * 7 + 3) % W
            free = (state.wall_map[new_r, new_c] == 0) & cycle & (state.time > 0)
            goal_id = scope_config["type_ids"]["goal"]
            otm = xp.where(free, set_at_2d(state.object_type_map, new_r, new_c, goal_id),
                           state.object_type_map)
            return dataclasses.replace(state, object_type_map=otm)
        return tick

    @classmethod
    def extra_state_schema(cls):
        return {"goals_collected": {"shape": ("n_agents",), "dtype": "int32"}}

    @classmethod
    def extra_state_builder(cls):
        def builder(parsed_arrays, scope):
            import numpy as np
            return {f"{scope}.goals_collected": np.zeros(2, dtype=np.int32)}
        return builder
```

The step pipeline calls all tick functions before movement and interactions each step.

**Interaction functions**

When an agent performs the interact action, the step pipeline evaluates branch functions in order. Each branch has the signature `(handled, ctx) -> (cond, updates, new_handled)`. The first branch whose `cond` is true claims the action via `handled`, preventing later branches from firing -- the handled cascade.

```python
def branch_collect_goal(handled, ctx):
    """Remove a goal from the grid when an agent faces it."""
    cond = ~handled & ctx["base_ok"] & (ctx["fwd_type"] == ctx["goal_id"])
    otm = set_at_2d(ctx["object_type_map"], ctx["fwd_r"], ctx["fwd_c"], 0)
    gc = ctx["goals_collected"]
    gc = gc.at[ctx["agent_idx"]].add(1) if hasattr(gc, "at") else gc
    return cond, {"object_type_map": otm, "goals_collected": gc}, handled | cond
```

Wire custom branches into an `interaction_fn` that builds a `ctx` dict from state arrays and static tables, runs each branch, and merges results via `merge_branch_results`. Put custom branches before generic ones like `branch_pickup` -- order determines priority.

```python
def build_goal_interaction_fn(scope):
    from cogrid.core.grid_object import object_to_idx
    goal_id = object_to_idx("goal", scope=scope)
    def interaction_fn(state, agent_idx, fwd_r, fwd_c, base_ok, scope_config):
        st = scope_config.get("static_tables", {})
        gc_key, gc = f"{scope}.goals_collected", state.extra_state.get(
            f"{scope}.goals_collected", xp.zeros(2, dtype=xp.int32))
        ctx = {"base_ok": base_ok, "fwd_r": fwd_r, "fwd_c": fwd_c, "agent_idx": agent_idx,
               "fwd_type": state.object_type_map[fwd_r, fwd_c], "goal_id": goal_id,
               "inv_item": state.agent_inv[agent_idx, 0], "goals_collected": gc,
               "agent_inv": state.agent_inv, "object_type_map": state.object_type_map,
               "object_state_map": state.object_state_map, "CAN_PICKUP": st["CAN_PICKUP"],
               "PICKUP_FROM_GUARD": st["PICKUP_FROM_GUARD"],
               "PLACE_ON_GUARD": st["PLACE_ON_GUARD"],
               "pickup_from_produces": st["pickup_from_produces"]}
        handled, results = xp.bool_(False), []
        for fn in [branch_collect_goal, branch_pickup, branch_drop_on_empty]:
            c, u, handled = fn(handled, ctx)
            results.append((c, u))
        m = merge_branch_results(results, {"agent_inv": state.agent_inv,
            "object_type_map": state.object_type_map, "object_state_map": state.object_state_map})
        extra = dict(state.extra_state)
        for c, u in results:
            if "goals_collected" in u:
                extra[gc_key] = xp.where(c, u["goals_collected"], gc)
        return dataclasses.replace(state, **m, extra_state=extra)
    return interaction_fn
```

The `handled` flag ensures each agent does at most one thing per interact action -- the first matching branch wins.

**Extra state**

`EnvState.extra_state` is a dict of scope-prefixed arrays for domain-specific data. Components declare `extra_state_schema` and `extra_state_builder` classmethods (shown on Goal above) -- the layout parser calls the builder at init time, and tick/interaction functions read and write these arrays at step time.

Extra state keys are prefixed with the scope name (e.g., `global.goals_collected`) and accessible as `state.goals_collected` in feature functions via StateView.

**Feature extractors**

Subclass `Feature`, decorate with `@register_feature_type`, set `per_agent` and `obs_dim`, and implement `build_feature_fn(cls, scope)` returning a `fn(state, agent_idx) -> ndarray` closure. The autowire system composes all registered features into a single observation function.

```python
@register_feature_type("goals_collected", scope="global")
class GoalsCollected(Feature):
    per_agent = True
    obs_dim = 1
    @classmethod
    def build_feature_fn(cls, scope):
        def fn(state, agent_idx):
            return xp.array([state.goals_collected[agent_idx]], dtype=xp.float32)
        return fn
```

Add `"goals_collected"` to the features list in the config. Features listed in the config are composed into a single `(obs_dim,)` vector per agent at init time.

**Run it**

```python
class GoalReward(InteractionReward):
    action = None
    overlaps = "goal"
layouts.register_layout("goal_v0", [
    "#######", "#  g  #", "# # # #",
    "#     #", "# # # #", "#+ + +#", "#######"])
def goal_terminated(prev_state, state, reward_config):
    goal_id = reward_config["type_ids"].get("goal", -1)
    rows, cols = state.agent_pos[:, 0], state.agent_pos[:, 1]
    return state.object_type_map[rows, cols] == goal_id
goal_config = {
    "name": "goal_finding", "num_agents": 2, "action_set": "cardinal_actions",
    "features": ["agent_dir", "agent_position", "can_move_direction", "inventory", "goals_collected"],
    "rewards": [GoalReward(coefficient=1.0, common_reward=True)],
    "grid": {"layout": "goal_v0"}, "max_steps": 50, "scope": "global",
    "terminated_fn": goal_terminated, "interaction_fn": build_goal_interaction_fn("global"),
}
registry.register("GoalFinding-V1", functools.partial(CoGridEnv, config=goal_config))
```

=== "NumPy"

    ```python
    env = registry.make("GoalFinding-V1")
    obs, info = env.reset(seed=42)
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminateds, truncateds, info = env.step(actions)
        print(rewards)
    ```

=== "JAX"

    ```python
    import jax
    env = registry.make("GoalFinding-V1", backend="jax")
    env.reset(seed=0)
    state, obs = env.jax_reset(jax.random.key(0))
    actions = jax.numpy.array([0, 3], dtype=jax.numpy.int32)
    state, obs, rewards, terminateds, truncateds, info = env.jax_step(state, actions)
    print(rewards)
    ```

For a full-featured example of these patterns, see the [Overcooked environment source](https://github.com/chasemcd/cogrid/tree/main/cogrid/envs/overcooked).
