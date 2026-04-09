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


@dataclasses.dataclass(frozen=True)
class ActionID:
    """Integer indices for each action in the current action set.

    Built at init time via :func:`build_action_id`.  Actions not present
    in the set have index ``-1`` (never matches any agent action).
    """

    move_up: int = -1
    move_down: int = -1
    move_left: int = -1
    move_right: int = -1
    forward: int = -1
    rotate_left: int = -1
    rotate_right: int = -1
    pickup_drop: int = -1
    toggle: int = -1
    noop: int = -1


_ACTION_NAME_TO_FIELD = {
    Actions.MoveUp: "move_up",
    Actions.MoveDown: "move_down",
    Actions.MoveLeft: "move_left",
    Actions.MoveRight: "move_right",
    Actions.Forward: "forward",
    Actions.RotateLeft: "rotate_left",
    Actions.RotateRight: "rotate_right",
    Actions.PickupDrop: "pickup_drop",
    Actions.Toggle: "toggle",
    Actions.Noop: "noop",
}


def build_action_id(action_set: tuple[str, ...]) -> ActionID:
    """Build an :class:`ActionID` from an action set tuple.

    Maps each action name in *action_set* to its positional index.
    """
    kwargs = {}
    for idx, action_name in enumerate(action_set):
        field = _ACTION_NAME_TO_FIELD[action_name]
        kwargs[field] = idx
    return ActionID(**kwargs)


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
