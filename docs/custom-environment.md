# Custom Environment

Build a goal-reaching grid world from scratch.

```
#######
#  g  #
# # # #
#     #
# # # #
#+ + +#
#######
```

```python
from cogrid.core.grid_object import GridObj, register_object_type, when
from cogrid.core.constants import Colors
from cogrid.core import layouts
from cogrid.core.rewards import InteractionReward
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs import registry
import functools
```

**Define a grid object**

Subclass `GridObj`, decorate with `@register_object_type` to add it to the scope registry, and declare its capabilities.

```python
@register_object_type("goal")
class Goal(GridObj):
    color = Colors.Green
    char = "g"
    can_overlap = when()  # agents can walk onto this cell

    def __init__(self, **kwargs):
        super().__init__(state=0)
```

Objects registered in the `"global"` scope (the default) are available to all environments.

**Define a reward**

Subclass `InteractionReward` with declarative conditions -- no manual computation needed.

```python
class GoalReward(InteractionReward):
    action = None        # no action required
    overlaps = "goal"    # fires when agent stands on a goal cell
```

The step pipeline sums all reward instances each step -- no manual composition needed.

**Register a layout**

ASCII strings map characters to registered objects: `#` = wall, `+` = spawn point, space = empty.

```python
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
```

`CoGridEnv` reads the layout and scope to auto-discover registered objects, rewards, and features at init time.

**Build the config and register**

```python
def goal_terminated(prev_state, state, reward_config):
    """Terminate when any agent reaches the goal."""
    goal_id = reward_config["type_ids"].get("goal", -1)
    rows, cols = state.agent_pos[:, 0], state.agent_pos[:, 1]
    return state.object_type_map[rows, cols] == goal_id

goal_config = {
    "name": "goal_finding",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": ["agent_dir", "agent_position", "can_move_direction", "inventory"],
    "rewards": [GoalReward(coefficient=1.0, common_reward=True)],
    "grid": {"layout": "goal_v0"},
    "max_steps": 50,
    "scope": "global",
    "terminated_fn": goal_terminated,
}

registry.register(
    "GoalFinding-V0",
    functools.partial(CoGridEnv, config=goal_config),
)
```

**Run it**

=== "NumPy"

    ```python
    from cogrid.envs import registry

    env = registry.make("GoalFinding-V0")
    obs, info = env.reset(seed=42)

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminateds, truncateds, info = env.step(actions)
    ```

=== "JAX"

    ```python
    import jax
    from cogrid.envs import registry

    env = registry.make("GoalFinding-V0", backend="jax")
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
        return all_rewards

    rewards = rollout(jax.random.key(0))
    ```

For a full-featured example of these patterns, see the [Overcooked environment source](https://github.com/chasemcd/cogrid/tree/main/cogrid/envs/overcooked).

!!! tip ""
    **Next: [Advanced Patterns](advanced-patterns.md)**
