# Overcooked

## Overview

Overcooked is a cooperative multi-agent cooking environment based on
[Carroll et al. (2019)](https://arxiv.org/abs/1910.05789). Two agents share a
kitchen and must coordinate to pick up ingredients, cook soups in pots, plate
the finished dishes, and deliver them to a serving area for reward.

![Overcooked Cramped Room](../assets/images/overcooked_grid.png){ width="60%" }

## Environment Details

### Game Mechanics

1. **Pick up ingredients** -- agents take onions (or tomatoes) from infinite
   supply stacks.
2. **Place in pot** -- ingredients are placed into a pot (capacity 3, same
   ingredient type per pot).
3. **Wait for cooking** -- once the pot is full, a 30-step cooking timer starts.
4. **Plate the soup** -- when the timer reaches zero the soup is ready; an agent
   holding a plate can pick it up from the pot.
5. **Deliver** -- carry the plated soup to a delivery zone to earn the delivery
   reward.

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

### Rewards

| Reward | Coefficient | Scope | Description |
|--------|-------------|-------|-------------|
| Delivery | 1.0 | Common | Deliver a plated soup to a delivery zone |
| Onion-in-pot | 0.1 | Individual | Place an onion into a pot with capacity |
| Soup-in-dish | 0.3 | Individual | Pick up a finished soup from a pot with a plate |

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
