# Implementing a New Environment

This guide walks through every step required to implement a new environment
with CoGrid, using the built-in Overcooked environment as a concrete reference.
By the end you will have:

- Grid objects with interaction properties
- A tick function for per-step state updates
- Custom interaction logic
- Reward functions
- Feature extractors for observations
- Layouts and environment registration

## Architecture Overview

Every CoGrid environment is assembled from four kinds of registered components:

```text
+-----------------------------+
|        CoGridEnv            |   Stateful wrapper (PettingZoo API)
+-----------------------------+
| config dict                 |   Declares scope, features, rewards, layout
+----+------------------------+
     |
     v
+----+------------------------+
|      Autowire Layer         |   Composes registered components into
|  build_scope_config()       |   a scope_config dict at init time
|  build_reward_config()      |
|  build_feature_config()     |
+----+------------------------+
     |
     v
+----+------------------------+
|      Step Pipeline          |   Pure function: (state, actions) -> (state, obs, rewards, ...)
|  tick -> move -> interact   |
|  -> observe -> reward       |
+-----------------------------+
```

Components are registered via decorators and discovered automatically by the
autowire layer. All custom behavior lives in your own project code and is
injected into the engine through decorators and config entries. The decorators
register your components into CoGrid's global registries at import time, so the
engine can discover them without any changes to CoGrid's source.


## Step 1: Create Your Project Structure

After installing CoGrid (`pip install cogrid`), create a Python package for
your environment in your own project:

```text
my_project/
    my_env/
        __init__.py               # Triggers decorator registration
        grid_objects.py           # Grid objects (@register_object_type)
        config.py           # Custom interaction fn, extra state builders (if needed)
        rewards.py                # Reward classes (Reward subclasses)
        features.py               # Feature extractors (@register_feature_type)
        agent.py                  # Custom Agent subclass (optional)
    train.py                      # Your training script
```

The `__init__.py` must import every module that contains `@register_*`
decorators so they execute at import time. This is what makes your components
visible to the CoGrid engine:

```python
# my_env/__init__.py
from my_env import grid_objects  # noqa: F401 -- triggers @register_object_type
from my_env import features      # noqa: F401 -- triggers @register_feature_type
```

!!! note
    You must `import my_env` (or the individual submodules) somewhere in your
    code **before** calling `registry.make()`. The decorators register
    components into CoGrid's global registries as a side effect of import.
    Rewards don't need import-time registration -- they are passed as
    instances in the config `"rewards"` list.


## Step 2: Define Grid Objects

Each object on the grid is a `GridObj` subclass registered with the
`@register_object_type` decorator.

**Required attributes:**

- `object_id`: Unique string identifier (e.g., `"onion"`).
- `color`: Rendering color (RGB tuple or `constants.Colors` enum).
- `char`: Single ASCII character for layout strings. Must be unique within
  the scope.

**Capability class attributes** declare static interaction properties that
the engine uses to build lookup tables. Use the `when()` descriptor (or
plain `True` for `is_wall`) as a class attribute:

| Attribute | Meaning |
|-----------|---------|
| `can_pickup = when()` | Agent can pick this up (removes from grid) |
| `can_overlap = when()` | Agent can walk onto this cell |
| `can_place_on = when()` | Agent can place held item onto this |
| `can_pickup_from = when()` | Agent can take an item from this (it stays) |
| `produces = "item_name"` | What a dispenser/stack creates on pickup_from |
| `consumes_on_place = True` | Item vanishes when placed on this (e.g., delivery zone) |
| `container = Container(...)` | Declares a stateful container (see below) |
| `recipes = [Recipe(...)]` | Recipes the container can cook (see below) |
| `is_wall = True` | Blocks movement |

The `scope` parameter on the decorator namespaces your objects so their
`char` and `object_id` values don't collide with objects from other
environments. Choose a unique scope string for your environment (e.g.,
`"my_env"`).

**Overcooked example** -- a simple pickupable object:

```python
# my_env/grid_objects.py
from cogrid.core import grid_object, constants
from cogrid.core.grid_object import register_object_type, when
from cogrid.visualization.rendering import fill_coords, point_in_circle

@register_object_type("onion", scope="overcooked")
class Onion(grid_object.GridObj):
    object_id = "onion"
    color = constants.Colors.Yellow
    char = "o"
    can_pickup = when()

    def render(self, tile_img):
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), self.color)
```

**Overcooked example** -- an infinite dispenser (stays on the grid, agent
receives a new item). Set the `produces` class attribute to the object ID
of the item dispensed:

```python
class _BaseStack(grid_object.GridObj):
    produces: str = None      # Object ID to dispense
    can_pickup_from = when()

@register_object_type("onion_stack", scope="overcooked")
class OnionStack(_BaseStack):
    color = constants.Colors.Yellow
    char = "O"
    produces = "onion"        # Agent receives a new Onion on pickup_from
```

**Overcooked example** -- a stateful container object (the Pot). For objects
that hold items, cook recipes, and produce outputs, use the `Container` and
`Recipe` declarative descriptors. The autowire system auto-generates all
array-level code (extra state, tick handler, interaction branches, static
tables, render sync):

```python
from cogrid.core.containers import Container, Recipe

@register_object_type("pot", scope="overcooked")
class Pot(grid_object.GridObj):
    color = constants.Colors.Grey
    char = "U"

    container = Container(capacity=3, pickup_requires="plate")
    recipes = [
        Recipe(["onion", "onion", "onion"], result="onion_soup", cook_time=30),
        Recipe(["tomato", "tomato", "tomato"], result="tomato_soup", cook_time=30),
    ]
    # Auto-generated by autowire:
    #   can_place_on = when(agent_holding=["onion", "tomato"])
    #   can_pickup_from = when(agent_holding="plate")
    #   extra_state_schema, extra_state_builder, tick_fn, render_sync, static_tables
```

For objects that need fully custom behavior beyond the Container/Recipe
pattern, use component classmethods instead (see Steps 3-5):

```python
@register_object_type("my_object", scope="my_env")
class MyObject(grid_object.GridObj):
    @classmethod
    def build_tick_fn(cls):
        """Return a (state, scope_config) -> state function."""
        ...

    @classmethod
    def extra_state_schema(cls):
        """Declare extra state arrays this object needs."""
        ...

    @classmethod
    def extra_state_builder(cls):
        """Return a function that initializes extra state."""
        ...

    @classmethod
    def build_static_tables(cls):
        """Return dict of additional lookup arrays."""
        ...

    @classmethod
    def build_render_sync_fn(cls):
        """Return a function that syncs array state to GridObj for rendering."""
        ...
```

!!! tip "Advanced: Generic Stacks and Factory Registration"
    If your environment has multiple dispenser types that share the same
    pick-up behavior (like Overcooked's ingredient stacks), you can use
    a shared base class and factory function instead of writing each one
    manually. The Overcooked environment provides `make_ingredient_and_stack()`
    which registers both an ingredient class and its stack class from a
    single call. See the [Overcooked docs](../environments/overcooked.md#custom-ingredients)
    for an example.


## Step 3: Declare Extra State (if needed)

The core `EnvState` tracks six standard arrays:

```text
agent_pos          (n_agents, 2)  int32   -- [row, col] per agent
agent_dir          (n_agents,)    int32   -- direction enum per agent
agent_inv          (n_agents, 1)  int32   -- held item type_id, -1 = empty
wall_map           (H, W)        int32   -- wall positions
object_type_map    (H, W)        int32   -- type_id at each cell
object_state_map   (H, W)        int32   -- object state at each cell
```

If your environment needs additional state (e.g., cooking timers, health
points), declare it via two classmethods on the relevant grid object:

**extra_state_schema** returns a dict of `{key: {"shape": tuple, "dtype": str}}`.
Shape elements can be strings like `"n_pots"` -- these are resolved at init
time by counting objects in the layout.

**extra_state_builder** returns a function that receives the parsed grid arrays
and initializes the extra state. It runs once at `reset()` time.

Extra state arrays are stored in `state.extra_state` with scope-prefixed
keys. The prefix is `"<scope>."` and is added automatically:

```python
# If your scope is "my_env" and you declared "timer" in the schema:
state.extra_state["my_env.timer"]   # (n_objects,) int32
```


## Step 4: Write the Tick Function (if needed)

The tick function runs once per step *before* movement and interactions. Use
it for time-based state machine logic (e.g., cooking timers, decay).

Signature: `(state: EnvState, scope_config: dict) -> EnvState`

```python
# my_env/config.py
import dataclasses
from cogrid.backend import xp

def my_tick_fn(state, scope_config):
    # Read extra state
    timer = state.extra_state["my_env.timer"]

    # Update (branchless for JAX compatibility)
    new_timer = xp.where(timer > 0, timer - 1, timer)

    # Return updated state
    new_extra = {**state.extra_state, "my_env.timer": new_timer}
    return dataclasses.replace(state, extra_state=new_extra)
```

Register it by returning it from `build_tick_fn` on a grid object class
(Step 2). Multiple objects can define `build_tick_fn` -- autowire composes
them into a single tick handler that calls each function sequentially.
Container/Recipe objects get an auto-generated tick function for free (timer
decrement), so you only need `build_tick_fn` for custom tick logic.


## Step 5: Write the Interaction Function (if needed)

The generic engine handles basic interactions automatically:

- **pickup**: Agent picks up an object from the cell ahead (if `can_pickup`).
- **pickup_from**: Agent receives an item from a dispenser (if `can_pickup_from`).
- **drop**: Agent drops held item onto an empty cell.
- **place_on**: Agent places held item onto a target (if `can_place_on`).

Additionally, the autowire system auto-generates interaction branches for:

- **Container/Recipe objects**: Placing ingredients into containers, picking up
  cooked results (e.g., the Pot). See Step 2.
- **Consume-on-place objects**: Items that vanish when placed (e.g.,
  `consumes_on_place = True` on DeliveryZone).

**Most environments do not need a custom interaction function.** The built-in
branches plus the Container/Recipe auto-generation cover pickup, drop, place,
container fill/cook/serve, and consume-on-place patterns.

If your environment needs interaction logic beyond these patterns (e.g.,
multi-step crafting, conditional state machine transitions), you can write
a custom interaction function and pass it via `config["interaction_fn"]`:

**Signature:**

```python
def my_interaction_fn(state, agent_idx, fwd_r, fwd_c, base_ok, scope_config):
    """
    Called once per agent per step (lower index = higher priority).

    Parameters
    ----------
    state : EnvState
        Current environment state.
    agent_idx : int
        Index of the acting agent.
    fwd_r, fwd_c : int
        Row and column of the cell the agent is facing.
    base_ok : bool
        True if no other agent occupies the forward cell.
    scope_config : dict
        Contains "type_ids", "static_tables", etc.

    Returns
    -------
    EnvState
        Updated state after the interaction.
    """
    ...
```

When `config["interaction_fn"]` is provided, it fully overrides the
auto-generated function. When omitted, autowire composes an interaction
function from the registered component capabilities.

**JAX compatibility**: All branches must compute results unconditionally using
`xp.where(condition, branch_result, fallback)` -- no Python `if/else` on
traced values. This allows JAX to compile the function.


## Step 6: Define Reward Functions

Reward functions are `Reward` subclasses. Each computes a `(n_agents,)`
float32 array. Parameters like coefficients are passed via `__init__` and
stored in `self.config`. Reward instances are listed explicitly in the env
config `"rewards"` key -- no decorator or auto-discovery needed.

### Pattern 1: Declarative interaction rewards (preferred)

Most rewards follow a pattern: check an action, check what the agent holds,
check what's ahead or underfoot, and apply a coefficient. The
`InteractionReward` base class handles this declaratively -- just set class
attributes for the conditions that must all be true:

```python
# Simple interaction -- just declare conditions:
from cogrid.core.rewards import InteractionReward

class OnionInPotReward(InteractionReward):
    action = "pickup_drop"
    holds = "onion"
    faces = "pot"
```

Available class attributes (all conditions are AND'd, `None` means "don't check"):

| Attribute | Type | Default | Meaning |
|-----------|------|---------|---------|
| `action` | `str` or `None` | **(required)** | `"pickup_drop"`, `"toggle"`, or `None` (no action filter) |
| `holds` | `str` or `None` | `None` | Type name agent must hold |
| `faces` | `str` or `None` | `None` | Type name in the forward cell |
| `overlaps` | `str` or `None` | `None` | Type name agent must stand on |
| `direction` | `int` or `None` | `None` | Direction agent must face (0=R,1=D,2=L,3=U) |

Instance config (`coefficient`, `common_reward`) is passed via `__init__` kwargs.

Override `extra_condition()` for domain-specific checks (pot capacity, timers):

```python
class OnionInPotReward(InteractionReward):
    action = "pickup_drop"
    holds = "onion"
    faces = "pot"

    def extra_condition(self, mask, prev_state, fwd_r, fwd_c, reward_config):
        # ... check pot capacity, type compatibility ...
        return mask & has_capacity & compatible
```

### Pattern 2: Position-based rewards (no action needed)

```python
class GoalReward(InteractionReward):
    action = None
    overlaps = "goal"
```

### Pattern 3: Complex rewards (use Reward directly)

For rewards that don't fit the declarative pattern (e.g., state-diff based
penalties, per-recipe lookups, multi-table joins), subclass `Reward` directly:

```python
from cogrid.core.rewards import Reward

class DeliveryReward(Reward):
    def compute(self, prev_state, state, actions, reward_config):
        """
        Parameters
        ----------
        prev_state : StateView
            State before the step. Dot-access for core fields
            (agent_pos, agent_inv, object_type_map, ...) and
            __getattr__ fallthrough for extra state (e.g., pot_timer).
        state : StateView
            State after the step.
        actions : ndarray
            (n_agents,) int32 action indices.
        reward_config : dict
            Contains "type_ids", "n_agents", "action_pickup_drop_idx",
            "action_toggle_idx".

        Returns
        -------
        ndarray
            (n_agents,) float32 final rewards.
        """
        type_ids = reward_config["type_ids"]
        n_agents = reward_config["n_agents"]
        coefficient = self.config.get("coefficient", 1.0)

        # Check which agents delivered soup
        holds_soup = prev_state.agent_inv[:, 0] == type_ids["onion_soup"]
        # ... compute forward positions, check delivery zone ...

        # Shared reward: broadcast sum to all agents
        n_earners = xp.sum(earns_reward.astype(xp.float32))
        return xp.full(n_agents, n_earners * coefficient, dtype=xp.float32)
```

Then add reward instances to the env config:

```python
from my_env.rewards import OnionInPotReward, DeliveryReward

my_config = {
    ...
    "rewards": [
        OnionInPotReward(coefficient=0.1),
        DeliveryReward(coefficient=1.0, common_reward=True),
    ],
}
```

**Key details:**

- `compute()` returns *final* `(n_agents,)` rewards. Apply any scaling or
  broadcasting (e.g., shared reward across all agents) inside `compute()`.
- Parameters (coefficients, flags) are passed via `__init__(**kwargs)` and
  accessed via `self.config`.
- `prev_state` and `state` are `StateView` objects, not `EnvState`.
  `StateView` provides dot-access with scope-stripped extra state keys
  (e.g., `prev_state.pot_timer` instead of
  `prev_state.extra_state["my_env.pot_timer"]`).
- Multiple reward instances can be listed in config. They are summed at step time.

!!! tip "Advanced: Static Tables in Rewards"
    If your reward logic needs environment-specific lookup data (e.g.,
    per-recipe reward values, deliverable item sets), access it via
    `reward_config["static_tables"]`. Static tables are built at init
    time by `build_static_tables()` classmethods on grid objects and
    flow through the autowire layer into reward_config automatically.
    See the [Overcooked docs](../environments/overcooked.md#recipe-system)
    for how per-recipe rewards use this pattern.


## Step 7: Define Feature Extractors

Features define the observation space. Each is a `Feature` subclass
registered with `@register_feature_type`.

```python
# my_env/features.py
from cogrid.backend import xp
from cogrid.core.features import Feature, register_feature_type

@register_feature_type("my_inventory", scope="my_env")
class MyInventory(Feature):
    per_agent = True   # True: fn(state, agent_idx), False: fn(state)
    obs_dim = 5        # Output size after ravel()

    @classmethod
    def build_feature_fn(cls, scope):
        """Build and return a pure feature function.

        Called once at init time. The returned function is called every
        step for every agent.
        """
        # Pre-compute constants at build time (not per-call)
        from cogrid.core.grid_object import object_to_idx
        item_type_ids = xp.array([
            object_to_idx("item_a", scope=scope),
            object_to_idx("item_b", scope=scope),
            object_to_idx("item_c", scope=scope),
            object_to_idx("item_d", scope=scope),
            object_to_idx("item_e", scope=scope),
        ], dtype=xp.int32)

        def fn(state, agent_idx):
            held = state.agent_inv[agent_idx, 0]
            return (item_type_ids == held).astype(xp.int32)

        return fn
```

**Two types of features:**

1. **Per-agent** (`per_agent = True`): Function signature is
   `fn(state, agent_idx) -> (obs_dim,) array`. Composed in ego-centric
   order: focal agent first, then other agents in ascending index.

2. **Global** (`per_agent = False`): Function signature is
   `fn(state) -> (obs_dim,) array`. Appended once after all per-agent blocks.

**Global features** (`per_agent = False`) are useful for encoding
environment-wide state that all agents share, such as active orders or
global timers. The Overcooked `OrderObservation` feature is an example:
it encodes the current order queue state identically for every agent.
See the [Overcooked docs](../environments/overcooked.md#order-observations)
for details.

**Composition order** (automatic):

```text
[focal per-agent feats] [agent_1 per-agent feats] ... [global feats]
```

**State access**: Like rewards, the `state` parameter in feature functions is
a `StateView`, so extra state is accessible via dot notation
(`state.pot_timer`).

**Layout hooks** (optional): If features depend on a layout index (e.g.,
one-hot layout encoding), register layout indices and a pre-compose hook:

```python
from cogrid.core.component_registry import (
    register_layout_indices,
    register_pre_compose_hook,
)

register_layout_indices("my_env", {
    "my_layout_v0": 0,
    "my_layout_v1": 1,
})

def _pre_compose_hook(layout_idx, scope):
    """Called before feature composition with the current layout index."""
    ...

register_pre_compose_hook("my_env", _pre_compose_hook)
```


## Step 8: Register Layouts and Environments

**Layouts** are ASCII strings where each character maps to a registered
grid object via its `char` attribute. Special characters:

```text
#   Wall (global scope)
+   Agent spawn point
C   Counter (global scope)
    (space) Empty / free space
```

All other characters are resolved from the scope you specify in the config.

Register a layout:

```python
from cogrid.core import layouts

layouts.register_layout(
    "my_layout_v0",
    [
        "#######",
        "#++  G#",
        "#     #",
        "#  X  #",
        "#G   G#",
        "#######",
    ],
)
```

**Environment config dict** -- declares everything the engine needs:

```python
my_config = {
    "name": "my_env",                        # Human-readable name
    "num_agents": 2,                          # Number of agents
    "action_set": "cardinal_actions",         # "cardinal_actions" or "rotation_actions"
    "features": [                             # Feature names to compose into obs
        "agent_dir",                          # Built-in feature (global scope)
        "agent_position",                     # Built-in feature (global scope)
        "my_inventory",                       # Your custom feature (Step 7)
    ],
    "grid": {"layout": "my_layout_v0"},       # Layout to use
    "max_steps": 1000,                        # Truncation limit
    "scope": "my_env",                        # Scope for component lookup
}
```

**Config keys reference:**

| Key | Description |
|-----|-------------|
| `name` | Human-readable environment name. |
| `num_agents` | Number of agents to place in the environment. |
| `action_set` | `"cardinal_actions"` (up/down/left/right + pickup/toggle/noop) or `"rotation_actions"` (forward + rotate + pickup/toggle/noop). |
| `features` | List of registered feature names to compose into observations. |
| `rewards` | List of `Reward` instances (e.g. `[DeliveryReward(coefficient=1.0)]`). |
| `grid` | Either `{"layout": "name"}` or `{"layout_fn": callable}`. |
| `max_steps` | Steps before truncation. |
| `scope` | Scope string for all component lookups. |
| `interaction_fn` | Custom interaction function (optional -- overrides auto-generated one, see Step 5). |
| `terminated_fn` | Custom termination function (optional, see below). |

**Register the environment:**

```python
import functools
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs import registry
from my_env.agent import MyAgent  # Your custom agent class (optional)

registry.register(
    "MyEnv-V0",
    functools.partial(
        CoGridEnv,
        config=my_config,
        agent_class=MyAgent,  # Omit to use the default Agent class
    ),
)
```

**Multiple layouts** can share the same config via `copy.deepcopy`:

```python
import copy

alt_config = copy.deepcopy(my_config)
alt_config["grid"]["layout"] = "my_layout_v1"

registry.register(
    "MyEnv-AltLayout-V0",
    functools.partial(CoGridEnv, config=alt_config, agent_class=MyAgent),
)
```

You can put all layout and registration code in your `my_env/__init__.py` or
in a dedicated setup module -- just make sure it runs before you call
`registry.make()`.


## Step 9: Custom Agent Class (optional)

The base `Agent` class handles movement, inventory, and direction. Subclass
it only if you need additional agent state (e.g., role tracking).

```python
# my_env/agent.py
from cogrid.core.agent import Agent

class MyAgent(Agent):
    """Agent subclass with role support."""
```

Pass the custom agent class via the `agent_class` parameter when registering
the environment (Step 8).


## Step 10: Custom Termination (optional)

By default, episodes end when `max_steps` is reached (truncation). For
custom termination conditions (e.g., all targets collected), pass a
`terminated_fn` in the config:

```python
from cogrid.backend import xp

def my_terminated_fn(prev_state, state, reward_config):
    """
    Returns (n_agents,) bool array. True = agent is terminated.
    """
    # Example: terminate when no targets remain on the grid
    type_ids = reward_config["type_ids"]
    targets_remaining = xp.any(state.object_type_map == type_ids["target"])
    all_done = ~targets_remaining
    return xp.full(state.n_agents, all_done, dtype=xp.bool_)

my_config["terminated_fn"] = my_terminated_fn
```


## Step 11: Use the Environment

```python
# train.py
from cogrid.envs import registry

import my_env  # Triggers all @register_* decorators in your package

env = registry.make("MyEnv-V0")
obs, info = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminateds, truncateds, info = env.step(actions)
```

The critical line is `import my_env` -- this executes your `__init__.py`
which imports the submodules, which triggers all the `@register_*` decorators
that register your grid objects, rewards, and features into CoGrid's engine.


## Complete Checklist

```text
[ ] 1. pip install cogrid
[ ] 2. Create my_env/ package in your project
[ ] 3. Write __init__.py with imports that trigger all @register_* decorators
[ ] 4. Define grid objects with @register_object_type (unique chars per scope)
[ ] 5. If needed: declare extra_state_schema + extra_state_builder on a grid object
[ ] 6. If needed: write tick function and register via build_tick_fn classmethod
[ ] 7. If needed: write interaction function (Container/Recipe handles most cases)
[ ] 8. Define Reward subclasses and add instances to config["rewards"]
[ ] 9. Define features with @register_feature_type
[ ] 10. Register layouts with layouts.register_layout()
[ ] 11. Build config dict and register environment with registry.register()
[ ] 12. If needed: subclass Agent for custom can_pickup logic
[ ] 13. If needed: add terminated_fn to config for custom episode termination
[ ] 14. import my_env before calling registry.make()
[ ] 15. Test: env = registry.make("MyEnv-V0"); env.reset()
```


## Component Reference

### Declarative Descriptors (preferred for stateful objects)

For objects that hold items, cook recipes, and produce outputs, use class
attributes instead of classmethods. The autowire system auto-generates all
the array-level code:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `container = Container(capacity, pickup_requires)` | `Container` | Declares a stateful container |
| `recipes = [Recipe(...)]` | `list[Recipe]` | Recipes the container can cook |
| `produces = "item_name"` | `str` | What a dispenser creates on pickup_from |
| `consumes_on_place = True` | `bool` | Item vanishes when placed (e.g., delivery zone) |

### Component Classmethods (for fully custom behavior)

These classmethods can be defined on any `@register_object_type` class. They
are discovered automatically by the decorator and wired into the step pipeline.
Use these when the Container/Recipe pattern does not cover your needs.

| Classmethod | Signature of returned value | Purpose |
|-------------|---------------------------|---------|
| `build_tick_fn` | `(state, scope_config) -> state` | Per-step state updates |
| `extra_state_schema` | `-> dict` | Declare extra state arrays |
| `extra_state_builder` | `-> fn(parsed, scope) -> dict` | Initialize extra state |
| `build_static_tables` | `-> dict` | Additional lookup arrays |
| `build_render_sync_fn` | `-> fn(grid, state, scope) -> None` | Sync arrays to GridObj for rendering |
