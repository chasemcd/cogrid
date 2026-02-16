"""Action definitions and action set groupings."""

import dataclasses


@dataclasses.dataclass
class Actions:
    """Available agent actions (movement, interaction, no-op)."""

    # Common
    PickupDrop = "pick_up_or_drop"  # Pick up an object
    Toggle = "toggle"  # Interact with/toggle/activate an object
    Noop = "no-op"

    # Rotation
    Forward = "move_forward"  # Move forward
    RotateLeft = "rotate_left"  # Turn Left
    RotateRight = "rotate_right"  # Turn Right

    # Cardinal Movement
    MoveLeft = "move_left"
    MoveRight = "move_right"
    MoveUp = "move_up"
    MoveDown = "move_down"


@dataclasses.dataclass
class ActionSets:
    """Predefined tuples of actions for rotation and cardinal control modes."""

    RotationActions = (
        Actions.Forward,
        Actions.PickupDrop,
        Actions.Toggle,
        Actions.Noop,
        Actions.RotateLeft,
        Actions.RotateRight,
    )
    CardinalActions = (
        Actions.MoveUp,
        Actions.MoveDown,
        Actions.MoveLeft,
        Actions.MoveRight,
        Actions.PickupDrop,
        Actions.Toggle,
        Actions.Noop,
    )
