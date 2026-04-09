# Overcooked V2

Seven environments from [Gessler et al., 2025](https://arxiv.org/abs/2503.17821) that test coordination under asymmetric information and stochasticity. Each episode samples a hidden target recipe. One agent can observe the recipe through a nearby indicator; the other must infer it through communication or partner behavior.

Uses the same [actions](overcooked.md#actions) and cooking pipeline as [Overcooked V1](overcooked.md). Key differences: partial observability, stochastic recipe selection, open pots that accept any ingredient, and reward shaping that penalizes incorrect actions.

<figure markdown="span">
  ![OvercookedV2 Layouts](../assets/images/v2_layouts.png){ width="100%" }
</figure>

## Variants

| Environment ID | Category | Layout |
|----------------|----------|--------|
| `OvercookedV2-CrampedRoomIndicator-V0` | Indicator Only | 5x4 |
| `OvercookedV2-GroundedCoordSimple-V0` | Grounded Coordination | 8x5 |
| `OvercookedV2-GroundedCoordRing-V0` | Grounded Coordination | 9x9 |
| `OvercookedV2-TestTimeSimple-V0` | Test-Time Protocol | 8x5 |
| `OvercookedV2-TestTimeWide-V0` | Test-Time Protocol | 6x7 |
| `OvercookedV2-DemoCookSimple-V0` | Demo Cook | 11x5 |
| `OvercookedV2-DemoCookWide-V0` | Demo Cook | 11x6 |

## Coordination Categories

| Category | Button | Incorrect Penalty | How Recipe is Communicated |
|----------|:------:|:-----------------:|----------------------------|
| **Grounded Coordination** | Yes (-5 cost) | -20 | Button reveals recipe to partner for 10 steps |
| **Test-Time Protocol** | No | -20 | Agents must develop implicit signaling conventions |
| **Demo Cook** | No | None | One agent demonstrates the recipe through actions |

## Stochastic Recipe Selection

At reset, a target recipe is drawn uniformly from `["onion_soup", "tomato_soup"]`. After each correct delivery, the target is resampled. Agents cannot memorize a fixed strategy; they must read the recipe each episode and adapt.

## Partial Observability

All V2 environments use `local_view` with `local_view_radius=2`, producing a **5x5 agent-centered window**. Each agent sees only the grid cells within 2 steps of its position. This creates information asymmetry: an agent near the recipe indicator can read the target, while an agent elsewhere cannot.

## Recipe and Button Indicators

The `RecipeIndicator` (`R`) is a wall tile whose state encodes the current target recipe. It is always active — any agent whose view window covers it can read the recipe.

The `ButtonIndicator` (`L`) is inactive by default. An agent can toggle it (empty-handed, costs -5 reward) to reveal the target recipe at that tile for 10 steps, then it deactivates. This is the communication channel in Grounded Coordination layouts.

## OpenPot

The `OpenPot` (`u`) accepts any 3-ingredient combination — onion, tomato, broccoli, or mushroom in any mix. There is no validation at placement time. Correctness is checked only at delivery: correct soup = +20, incorrect = -20 (or 0 in Demo Cook). Cook time is 20 steps.

## V2-Specific Objects

In addition to [shared Overcooked objects](overcooked.md#objects):

| Char | Name | Description |
|------|------|-------------|
| `u` | OpenPot | Pot accepting any 3-ingredient combination (cook time 20) |
| `R` | RecipeIndicator | Displays the current target recipe (wall tile) |
| `L` | ButtonIndicator | Toggle to reveal recipe for 10 steps (-5 cost) |
| `X` | OpenDeliveryZone | Accepts any soup type for delivery |
| `B` | BroccoliStack | Distractor ingredient dispenser |
| `M` | MushroomStack | Distractor ingredient dispenser |

## Observations

The local view is a `(5, 5, 35)` tensor (flattened to 875) with these channel groups:

| Channels | Count | Description |
|----------|:-----:|-------------|
| Core | 8 | Object type map, agent positions (2), directions (2), inventories (2), object state map |
| Pot state | 4 | Is cooking, is ready, fill level, cook timer (all at pot positions only) |
| Pot ingredients | 4 | Per-ingredient count in pot (onion, tomato, broccoli, mushroom), normalized by capacity |
| Decomposed inventory | 12 | `[plate, cooked, onion, tomato, broccoli, mushroom]` at each agent's position (self first, 6 channels per agent) |
| Recipe decomposition | 6 | `[plate, cooked, onion, tomato, broccoli, mushroom]` at RecipeIndicator and active ButtonIndicator positions |
| Delivery indicator | 1 | 1.0 at delivery zone positions on the step a delivery occurred |

Recipe decomposition channels are non-zero only at indicator tiles. The button's channels activate when its timer is running and return to zero when it expires.

## Rewards

| Class | Coefficient | Common | Trigger |
|-------|-------------|--------|---------|
| `TargetRecipeDeliveryReward` | 20.0 | Yes | Correct delivery +20, incorrect -20 |
| `TargetRecipeIngredientInPotReward` | 3.0 | Yes | Correct ingredient in pot +3, incorrect -3 |
| `TargetRecipeSoupInDishReward` | 5.0 | Yes | Correct soup plated +5, incorrect -5 |
| `ButtonActivationCost` | -5.0 | Yes | Toggle the button indicator |

Shaped rewards (ingredient-in-pot and soup-in-dish) can be annealed to zero over training via `env.set_reward_coefficients()`.

## Layouts

### Cramped Room Indicator
```
CCuCC
O   T
=   R
CCCXC
```

### Grounded Coordination Simple
```
CCBCCCCC
C  C=  O
R +Lu+ X
C  C=  T
CCBCCCCC
```

### Grounded Coordination Ring
```
CCCBRBCCC
C       C
C CCLCC C
B O   = B
R+X+u + R
B T   = B
C CCLCC C
C       C
CCCBRBCCC
```

### Test-Time Protocol Simple
```
CCBCCCCC
C  C=  O
R +Cu+ X
C  C=  T
CCBCCCCC
```

### Test-Time Protocol Wide
```
CCX=CC
O +  O
T    T
CuCuCC
M +  M
C    C
CCRCCC
```

### Demo Cook Simple
```
CCCCCRBCoCC
O      C  =
C     +u+ X
T      C  =
CCCCCRBCtCC
```

### Demo Cook Wide
```
CCCC=X=CCCC
CCCO + TCCC
CCCCCuCCCCC
C    +    C
O  CMRMC  O
CTCCCCCCCTC
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_recipes` | `list[str]` | `["onion_soup", "tomato_soup"]` | Recipes the target is sampled from |
| `resample_on_delivery` | `bool` | `True` | Resample target recipe after correct delivery |
| `local_view_radius` | `int` | `2` | Observation window radius (5x5 at radius 2) |
| `max_steps` | `int` | `400` | Episode length |
