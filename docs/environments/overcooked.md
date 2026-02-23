# Overcooked

## Overview

Overcooked is a cooperative multi-agent cooking environment based on
[Carroll et al. (2019)](https://arxiv.org/abs/1910.05789). Two agents share a
kitchen and must coordinate to pick up ingredients, cook soups in pots, plate
the finished dishes, and deliver them to a serving area for reward.

![Overcooked Cramped Room](../assets/images/overcooked_grid.png){ width="60%" }

## Environment Details

### Game Mechanics

1. **Pick up ingredients** -- agents take ingredients (onions, tomatoes, or
   custom types) from infinite supply stacks.
2. **Place in pot** -- ingredients are placed into a pot (capacity 3). The pot
   accepts any combination of ingredients that matches a valid recipe prefix.
3. **Wait for cooking** -- once the pot is full and matches a recipe, a
   per-recipe cooking timer starts (default 30 steps for the built-in recipes).
4. **Plate the soup** -- when the timer reaches zero the soup is ready; an agent
   holding a plate can pick it up from the pot.
5. **Deliver** -- carry the plated soup to a delivery zone to earn the delivery
   reward.

By default, the classic onion soup and tomato soup recipes are used. See
[Recipe System](#recipe-system) below for custom recipes.

### Objects

| Object | Char | Description |
|--------|------|-------------|
| OnionStack | `O` | Infinite supply of onions |
| TomatoStack | `T` | Infinite supply of tomatoes |
| Onion | `o` | A single onion ingredient |
| Tomato | `t` | A single tomato ingredient |
| Pot | `U` | Cooking pot (capacity 3, 30-step timer) |
| PlateStack | `=` | Infinite supply of plates |
| Plate | `P` | A plate for carrying soup |
| OnionSoup | `S` | Completed onion soup |
| TomatoSoup | `!` | Completed tomato soup |
| DeliveryZone | `@` | Delivery area for serving soup |
| Counter | `C` | Impassable counter surface |

This is the default set. Custom ingredients and their stacks can be registered
at init time using `make_ingredient_and_stack()` -- see
[Custom Ingredients](#custom-ingredients) below.

### Rewards

| Class | Coefficient | Scope | Description |
|-------|-------------|-------|-------------|
| `OnionSoupDeliveryReward` | 1.0 | Common | Simplest delivery: fires on pickup_drop + holds onion_soup + faces delivery_zone. Fixed coefficient, no recipe lookup. |
| `DeliveryReward` | 1.0 | Common | Multi-recipe delivery: uses IS_DELIVERABLE table and per-recipe reward values. Fires for any valid delivery (no order gating). |
| `OrderDeliveryReward` | 1.0 | Common | Extends `DeliveryReward` with order gating (fires only when a matching active order is consumed) and optional tip bonus proportional to remaining order time. |
| `OnionInPotReward` | 0.1 | Individual | Place an onion into a pot with capacity (pickup_drop + holds onion + faces pot). |
| `SoupInDishReward` | 0.3 | Individual | Pick up a finished soup from a pot with a plate (pickup_drop + holds plate + faces pot). |
| `ExpiredOrderPenalty` | -5.0 | Common | Penalty when an active order expires (requires order queue). |

The default `cramped_room_config` uses `DeliveryReward` (no order gating).
To enable order-based rewards (order gating, tip bonus, expired penalty),
use `OrderDeliveryReward` and configure the order queue -- see
[Order Queue](#order-queue) below.

## Available Layouts

| Environment ID | Layout | Description |
|----------------|--------|-------------|
| `Overcooked-CrampedRoom-V0` | Cramped Room | Small 5x5 kitchen, tight coordination required |
| `Overcooked-AsymmetricAdvantages-V0` | Asymmetric Advantages | Asymmetric access to ingredients and delivery |
| `Overcooked-CoordinationRing-V0` | Coordination Ring | Circular layout requiring movement coordination |
| `Overcooked-ForcedCoordination-V0` | Forced Coordination | Agents have fixed spawn points and must pass items |
| `Overcooked-CounterCircuit-V0` | Counter Circuit | Large layout with central counter island |
| `Overcooked-RandomizedLayout-V0` | Randomized | Randomly selects one of the five layouts each episode |
| `Overcooked-CrampedRoom-SingleAgent-V0` | Cramped Room (1 agent) | Single-agent variant of Cramped Room |

## Recipe System

By default, Overcooked uses two built-in recipes -- onion soup and tomato soup.
Recipes are defined as a list of dictionaries, each specifying the ingredients,
result, cook time, and reward value:

```python
DEFAULT_RECIPES = [
    {
        "ingredients": ["onion", "onion", "onion"],
        "result": "onion_soup",
        "cook_time": 30,
        "reward": 20.0,
    },
    {
        "ingredients": ["tomato", "tomato", "tomato"],
        "result": "tomato_soup",
        "cook_time": 30,
        "reward": 20.0,
    },
]
```

Each recipe has four keys:

- **`ingredients`** -- list of ingredient names (must be registered object types). Recipes can use mixed ingredients (e.g., `["onion", "onion", "tomato"]`).
- **`result`** -- the name of the output object produced when the recipe is completed.
- **`cook_time`** -- number of environment steps the pot takes to cook once full.
- **`reward`** -- the delivery reward value for this recipe's output.

At init time, `compile_recipes()` compiles the recipe list into fixed-shape
lookup arrays that the interaction system uses for recipe matching during
gameplay.

!!! note "Config wiring status"
    The recipe infrastructure supports arbitrary recipes, including
    mixed-ingredient combinations. Full config-dict wiring (selecting recipes
    via the environment config) is planned; currently, custom recipes require
    calling `compile_recipes()` directly.

## Order Queue

By default, the order queue is disabled and delivery rewards work as in the
original Overcooked -- any valid delivery earns a reward regardless of timing.

When enabled, the order queue adds a timed order lifecycle:

1. **Spawn** -- new orders appear at regular intervals, each requesting a specific recipe.
2. **Countdown** -- each active order counts down every step.
3. **Deliver** -- delivering the correct soup consumes the matching order and earns the delivery reward.
4. **Expire** -- if the countdown reaches zero, the order expires and triggers the `ExpiredOrderPenalty`.

The order queue is configured with a dictionary:

```python
order_config = {
    "max_active": 3,          # Maximum concurrent orders
    "spawn_interval": 40,     # Steps between new order spawns
    "time_limit": 200,        # Steps before an order expires
    "recipe_weights": [2.0, 1.0],  # Relative spawn frequency per recipe
}
```

- **`max_active`** -- the maximum number of orders that can be active simultaneously.
- **`spawn_interval`** -- how many steps to wait between spawning new orders.
- **`time_limit`** -- how many steps an order lasts before it expires.
- **`recipe_weights`** -- relative weights controlling how often each recipe is requested (uses deterministic round-robin, not random sampling).

!!! note "Config wiring status"
    The order queue infrastructure is complete. Full config-dict wiring is
    planned; currently, enabling orders requires custom
    `build_overcooked_extra_state` calls with an `order_config` dict.

## Custom Ingredients

New ingredient types and their corresponding stacks can be registered at
runtime using `make_ingredient_and_stack()`. This must be called **before**
environment creation so the new types are available when the interaction
tables are built.

```python
from cogrid.envs.overcooked.overcooked_grid_objects import make_ingredient_and_stack
from cogrid.core import constants

# Must be called BEFORE environment creation
Mushroom, MushroomStack = make_ingredient_and_stack(
    ingredient_name="mushroom",
    ingredient_char="m",
    ingredient_color=constants.Colors.Brown,
    stack_name="mushroom_stack",
    stack_char="M",
    scope="overcooked",
)
```

The factory function creates both classes, registers them in the object
registry, and returns the class references. The new ingredient can then be
used in custom recipes, and the new stack can be placed in layout grids.

Custom ingredients are automatically reflected in `OvercookedInventory`
observations -- the feature dynamically discovers all pickupable types from
the registry at compose time, so `obs_dim` adjusts to include any new
ingredient types.

## Order Observations

The `OrderObservation` feature encodes the current state of active orders into
the observation vector. Each active order is represented as a recipe type
(one-hot encoded) plus a normalized time remaining value.

- **Observation dimension:** `max_active * (n_recipes + 1)` (default: 3 orders x 3 values = 9).
- **Per order:** `n_recipes` values for the recipe one-hot, plus 1 value for normalized time remaining (0.0 to 1.0).
- **Global feature:** `per_agent = False` -- all agents receive the same order observation.
- **Backward compatible:** returns zeros when the order queue is not configured.

To include order observations in the feature set, add `"order_observation"` to
the features list in the environment config:

```python
config = {
    "features": [
        # ... existing features ...
        "order_observation",
    ],
}
```

## Quick Start

```python
from cogrid.envs import registry
import cogrid.envs.overcooked

env = registry.make("Overcooked-CrampedRoom-V0")
obs, info = env.reset(seed=42)

while env.agents:
    actions = {a: env.action_space.sample() for a in env.agents}
    obs, rewards, terminateds, truncateds, info = env.step(actions)
```

## Links

- [Custom Environment Tutorial](../tutorials/custom-environment.md) -- how
  Overcooked is built using the component API
- [JAX Backend Tutorial](../tutorials/jax-backend.md) -- training Overcooked
  agents with JAX and vmap
- [API Reference: cogrid.envs.overcooked](../reference/cogrid/envs/overcooked/index.md) --
  module documentation
