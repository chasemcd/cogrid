# Actions

Agents choose one action per step. CoGrid provides two action sets that determine how agents move and interact.

## Action Sets

The environment config selects an action set via `"action_set"`:

```python
config = {
    "action_set": "cardinal_actions",  # or "rotation_actions"
    ...
}
```

### Cardinal Actions

Direct four-directional movement. The agent always faces its last movement direction.

| Index | Action | Description |
|-------|--------|-------------|
| 0 | `MoveUp` | Move one cell up |
| 1 | `MoveDown` | Move one cell down |
| 2 | `MoveLeft` | Move one cell left |
| 3 | `MoveRight` | Move one cell right |
| 4 | `PickupDrop` | Pick up or drop an object |
| 5 | `Toggle` | Interact with / toggle / activate an object |
| 6 | `Noop` | Do nothing |

### Rotation Actions

Turn-and-advance movement. The agent has a facing direction and moves forward relative to it.

| Index | Action | Description |
|-------|--------|-------------|
| 0 | `Forward` | Move one cell in the facing direction |
| 1 | `PickupDrop` | Pick up or drop an object |
| 2 | `Toggle` | Interact with / toggle / activate an object |
| 3 | `Noop` | Do nothing |
| 4 | `RotateLeft` | Turn 90 degrees counter-clockwise |
| 5 | `RotateRight` | Turn 90 degrees clockwise |

## Movement

Movement actions attempt to place the agent in the target cell. The move succeeds if:

1. The target cell is within grid bounds.
2. The cell is not a wall (`wall_map[r, c] == 0`).
3. The object in the cell has `can_overlap` set (e.g. Floor).

If any condition fails, the agent stays in place.

## Interaction: PickupDrop

The `PickupDrop` action handles the pickup/drop pipeline based on the agent's inventory and the cell it faces:

| Agent Holds | Facing | Result |
|-------------|--------|--------|
| Nothing | Pickupable object | Pick up the object |
| Nothing | Container with `pickup_requires=None` | Pick up from container |
| Required item | Container with `pickup_requires` set | Pick up from container |
| Item | `can_place_on` surface (e.g. Counter) | Place item on surface |
| Ingredient | Container with capacity | Add ingredient to container |

For containers, adding the final ingredient starts the cook timer. When the timer reaches zero, the result item becomes available for pickup.

## Interaction: Toggle

The `Toggle` action activates objects that respond to toggling. For example, a closed Door opens when toggled, and a locked Door opens if the agent holds the matching Key.

## Action Definitions

Actions are defined in `cogrid.core.actions`:

```python
from cogrid.core.actions import Actions, ActionSets

# Individual action strings
Actions.MoveUp       # "move_up"
Actions.PickupDrop   # "pick_up_or_drop"
Actions.Toggle       # "toggle"
Actions.Noop         # "no-op"

# Action set tuples
ActionSets.CardinalActions   # (MoveUp, MoveDown, MoveLeft, MoveRight, PickupDrop, Toggle, Noop)
ActionSets.RotationActions   # (Forward, PickupDrop, Toggle, Noop, RotateLeft, RotateRight)
```

The action space exposed to agents is a `Discrete(n)` space where `n` is the length of the selected action set.
