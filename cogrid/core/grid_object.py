"""GridObj defines an object in the CoGridEnv environment. It is largely derived from the Minigrid WorldObj:
https://github.com/Farama-Foundation/Minigrid/minigrid/core/world_object.py

"""

from __future__ import annotations
from copy import deepcopy
import math
import uuid

import numpy as np


from cogrid.constants import GridConstants
from cogrid.core.constants import ObjectColors, Colors, COLORS, COLOR_NAMES
from cogrid.core import constants
from cogrid.core.directions import Directions
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
    add_text_to_image,
)


# The OBJECT_REGISTRY holds all registered objects under a "scope" (e.g., "global", "search_rescue", "overcooked")
# which allows us to re-use IDs and character representations across environments (e.g., P is a purple target in
# search_rescue and a plate in overcooked).
OBJECT_REGISTRY: dict[str, dict[str, GridObj]] = {}


def make_object(
    object_id: str | None, scope: str = "global", **kwargs
) -> GridObj:
    if object_id is None:
        return None

    if scope not in OBJECT_REGISTRY:
        raise ValueError(
            f"No objects registered with scope `{scope}`. Existing scopes are {list(OBJECT_REGISTRY.keys())}."
        )

    if object_id in OBJECT_REGISTRY["global"]:
        return OBJECT_REGISTRY["global"][object_id](**kwargs)
    elif object_id not in OBJECT_REGISTRY[scope]:
        raise ValueError(
            f"Object with object_id `{object_id}` not registered in scope `{scope}`. "
            f"Call register_object('{object_id}', <class>, scope='{scope}') to add it to the registry."
        )

    return OBJECT_REGISTRY[scope][object_id](**kwargs)


def get_object_class(object_id: str, scope: str = "global") -> GridObj:
    return OBJECT_REGISTRY[scope][object_id]


def register_object(
    object_id: str, obj_class: GridObj, scope: str = "global"
) -> None:
    global_scope_chars = [
        obj.char for obj in OBJECT_REGISTRY.get("global", {}).values()
    ]

    if obj_class.char in global_scope_chars:
        raise ValueError(
            f"Character `{obj_class.char}` is already in use in the global scope. "
            "Please choose a different character."
        )

    if object_id in OBJECT_REGISTRY.get("global", {}):
        raise ValueError(
            f"Object with object_id `{object_id}` already registered in the global scope. "
            "Please select a different ID."
        )

    if scope not in OBJECT_REGISTRY:
        OBJECT_REGISTRY[scope] = {}

    OBJECT_REGISTRY[scope][object_id] = obj_class


def get_registered_object_ids(scope: str = "global") -> list[str]:
    """Return a list of the object_ids of available objects in a given scope."""
    return list(OBJECT_REGISTRY[scope].keys())


def get_object_char(object_id: str, scope: str = "global") -> str:
    return get_object_class(object_id, scope=scope).char


def get_object_id_from_char(object_char: str, scope: str = "global") -> str:
    # First check global scope, no matter what scope was passed (default to global).
    for object_id, object_class in OBJECT_REGISTRY["global"].items():
        if object_class.char == object_char:
            return object_id

    if scope != "global":
        for object_id, object_class in OBJECT_REGISTRY[scope].items():
            if object_class.char == object_char:
                return object_id

    raise ValueError(
        f"There is no registered object with character representation `{object_char}` in scope `{scope}`."
    )


# TODO(chase): This is temporary and needs to be retrieved from the registry.
OBJECT_NAMES = [
    None,
    "free_space",
    "wall",
    "door",
    "key",
    "pickaxe",
    "medkit",
    "rubble",
    "green_victim",
    "purple_victim",
    "yellow_victim",
    "red_victim",
    "pot",
    "plate",
    "plate_stack",
    "onion_soup",
    "onion_stack",
    "onion",
    "delivery_zone",
    "counter",
] + [f"agent_{direction}" for direction in "^>v<"]


class GridObj:
    object_id: str = None
    color: str | tuple = None
    char: str = None

    def __init__(
        self,
        state: int = 0,
        toggle_value: float = 0,
        inventory_value: float = 0,
        overlap_value: float = 0,
        placed_on_value: float = 0,
        picked_up_from_value: float = 0,
    ):
        self.uuid: str = str(uuid.uuid4())

        self.state: int = state

        # If an object can be placed on top of this one, this will hold the object that's on top.
        self.obj_placed_on: GridObj | None = None

        # position info
        self.init_pos: tuple[int, int] | None = None
        self.pos: tuple[int, int] | None = None

        # defines rewards for holding/toggling/overlapping
        self.toggle_value: float | int = toggle_value
        self.inventory_value: float | int = inventory_value
        self.overlap_value: float | int = overlap_value
        self.placed_on_value: float | int = placed_on_value
        self.picked_up_from_value: float | int = picked_up_from_value

    def can_overlap(self, agent: GridAgent) -> bool:
        """Can an agent overlap with this object?"""
        return False

    def can_pickup(self, agent: GridAgent) -> bool:
        """Can an agent pick this object up and store in inventory?"""
        return False

    def can_place_on(self, agent: GridAgent, cell: GridObj) -> bool:
        """
        Can another object be placed on top of this object? e.g., a countertop that can't be walked through
        but can have an item on top of it.
        """
        return False

    def can_pickup_from(self, agent: GridAgent) -> bool:
        """Can the agent pick up an object from this one?"""
        return self.obj_placed_on is not None and self.obj_placed_on.can_pickup(
            agent=agent
        )

    def place_on(self, agent: GridAgent, cell: GridObj) -> None:
        self.obj_placed_on = cell
        self.state = object_to_idx(cell)

    def pick_up_from(self, agent: GridAgent) -> GridObj:
        assert (
            self.obj_placed_on is not None
        ), f"Picking up from but there's no object placed on {self.object_id}"
        cell = self.obj_placed_on
        self.obj_placed_on = None
        self.state = 0
        return cell

    def see_behind(self, agent: GridAgent) -> bool:
        """Can the agent see through this object?"""
        return True

    def visible(self) -> bool:
        return True

    def toggle(self, env, agent: GridAgent = None) -> bool:
        """
        Trigger/Toggle an action this object performs. Some toggles are conditioned on the environment
        and require specific conditions to be met, which can be checked with the end.
        """
        return False

    def encode(self, encode_char=True):
        return (
            self.char if encode_char else object_to_idx(self),
            0,  # TODO(chase): Remove. This used to be color, but we're no longer using it.
            int(self.state),
        )

    def render(self, tile_img):
        """By default, everything will be rendered as a square with the specified color."""
        fill_coords(tile_img, point_in_rect(0, 1, 0, 1), color=self.color)

    @staticmethod
    def decode(char_or_idx: str | int, state: int, scope: str = "global"):

        if char_or_idx in [
            None,
            GridConstants.FreeSpace,
            GridConstants.Obscured,
        ]:
            return None

        # check if the name was passed instead of the character
        if _is_str(char_or_idx) and len(char_or_idx) > 1:
            object_id = char_or_idx
        elif _is_str(char_or_idx):
            object_id = get_object_id_from_char(char_or_idx, scope=scope)
        else:
            raise ValueError(f"Invalid identifier for decoding: {char_or_idx}")

        state = int(state)

        return make_object(object_id, state=state, scope=scope)

    def rotate_left(self):
        """Some objects (e.g., agents) have a rotation and must be rotated with the grid."""
        pass

    def tick(self):
        """
        Some objects have a time component (e.g., cooking soup), so we call the tick
        method on all objects for each env.step()
        """
        pass

    def _remove_from_grid(self, grid):
        cell = grid.get(*self.pos)
        assert self is cell
        grid.set(*self.pos, None)


def _is_str(chk):
    return isinstance(chk, str) or isinstance(chk, np.str)


def _is_int(chk):
    return isinstance(chk, int) or isinstance(chk, np.int)


def object_to_idx(object: GridObj | str | None):
    if isinstance(object, GridObj):
        object_id = object.object_id
    else:
        object_id = object

    return OBJECT_NAMES.index(object_id)


def idx_to_object(idx: int):
    for name in OBJECT_NAMES:
        if OBJECT_NAMES.index(name) == idx:
            return name
    raise ValueError(f"Object index {idx} not in OBJECT_LIST.")


class GridAgent(GridObj):
    def __init__(self, agent):
        """
        Grid agents are initialized slightly differently. State corresponds to the object they are holding
        and char/colors are unique for each agent.
        """
        self.color = {
            1: ObjectColors.AgentOne,
            2: ObjectColors.AgentTwo,
            3: ObjectColors.AgentThree,
            4: ObjectColors.AgentFour,
        }[agent.agent_number]

        self.char = {
            Directions.Up: "^",
            Directions.Down: "v",
            Directions.Left: "<",
            Directions.Right: ">",
        }[agent.dir]

        assert (
            len(agent.inventory) <= 1
        ), "Current implementation requires maximum inventory size of 1."

        self.object_id = f"agent_{self.char}"

        # TODO(chase): State must encapsulate carried objects and role
        # state = agent.role_idx
        state = (
            0
            if len(agent.inventory) == 0
            else object_to_idx(agent.inventory[0])
        )

        super().__init__(state=state)
        self.dir = agent.dir
        self.pos = agent.pos
        self.front_pos = agent.front_pos
        self.agent_id = agent.id
        self.inventory: list[GridObj] = deepcopy(agent.inventory)
        assert self.pos is not None

    def rotate_left(self):
        self.char = {"^": "<", "<": "v", "v": ">", ">": "^"}[self.char]
        self.object_id = f"agent_{self.char}"
        self.dir -= 1
        if self.dir < 0:
            self.dir += 4

    def render(self, tile_img):
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the triangle based on agent direction
        assert self.dir is not None
        tri_fn = rotate_fn(
            tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir
        )
        fill_coords(tile_img, tri_fn, self.color)

        # add any item in the inventory to the corner
        inv_tile_rows, inv_tile_cols = (
            tile_img.shape[0] // 3,
            tile_img.shape[1] // 3,
        )
        assert (
            len(self.inventory) <= 3
        ), "We're rendering inventory items at 1/3 size, so can't do more than 3!"

        offset = 4  # offset so we still see grid lines
        for i, obj in enumerate(self.inventory):
            inventory_tile = np.zeros(shape=(inv_tile_rows, inv_tile_cols, 3))
            obj.render(inventory_tile)

            # Take the subset of the image that we'll fill, then only fill where the image is non-zero
            # this makes a transparent background, rather than adding in a black square
            tile_subset = tile_img[
                i * inv_tile_rows + offset : (i + 1) * inv_tile_rows + offset,
                offset : inv_tile_cols + offset,
                :,
            ]
            nonzero_entries = np.nonzero(inventory_tile)
            tile_subset[nonzero_entries] = inventory_tile[nonzero_entries]

    @staticmethod
    def decode(char_or_idx: str | int, state: int):

        if char_or_idx in [
            None,
            GridConstants.FreeSpace,
            GridConstants.Obscured,
        ]:
            return None

        # check if the name was passed instead of the character
        if _is_str(char_or_idx) and len(char_or_idx) > 1:
            object_id = char_or_idx
        elif _is_str(char_or_idx):
            object_id = get_object_id_from_char(char_or_idx)
        elif _is_int(char_or_idx):
            object_id = OBJECT_NAMES[char_or_idx]
        else:
            raise ValueError(f"Invalid identifier for decoding: {char_or_idx}")

        state = int(state)

        return make_object(object_id, state=state)


class Wall(GridObj):
    object_id = "wall"
    color = constants.Colors.Grey
    char = "#"

    def __init__(self, *args, **kwargs):
        super().__init__(state=0)

    def see_behind(self) -> bool:
        return False


register_object(Wall.object_id, Wall, scope="global")


class Floor(GridObj):
    object_id = "floor"
    color = constants.Colors.PaleBlue
    char = GridConstants.FreeSpace

    def __init__(self, **kwargs):
        super().__init__(
            state=0,
        )

    def can_overlap(self) -> bool:
        return True


register_object(Floor.object_id, Floor, scope="global")


class Counter(GridObj):
    object_id = "counter"
    color = constants.Colors.LightBrown
    char = "C"

    def __init__(self, state: int = 0, **kwargs):
        # TODO(chase): need to be able to initialize an object on top
        #   via the state. Take state and map it to an object.
        super().__init__(
            state=state,
        )

    def can_place_on(self, agent: GridAgent, cell: GridObj) -> bool:
        return self.state == 0

    def render(self, tile_img):
        super().render(tile_img)

        if self.obj_placed_on is not None:
            self.obj_placed_on.render(tile_img)


register_object(Counter.object_id, Counter, scope="global")


class Key(GridObj):
    object_id = "key"
    color = constants.Colors.Yellow
    char = "K"

    def __init__(self, state=0):
        super().__init__(state=state)

    def can_pickup(self, agent: GridAgent):
        return True

    def render(self, tile_img):
        # Vertical quad
        fill_coords(tile_img, point_in_rect(0.50, 0.63, 0.31, 0.88), self.color)

        # Teeth
        fill_coords(tile_img, point_in_rect(0.38, 0.50, 0.59, 0.66), self.color)
        fill_coords(tile_img, point_in_rect(0.38, 0.50, 0.81, 0.88), self.color)

        # Ring
        fill_coords(
            tile_img, point_in_circle(cx=0.56, cy=0.28, r=0.190), self.color
        )
        fill_coords(
            tile_img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0)
        )


register_object(Key.object_id, Key, scope="global")


class Door(GridObj):
    object_id = "door"
    color = constants.Colors.DarkGrey
    char = "D"

    def __init__(self, state):
        super().__init__(state=state)
        self.is_open = state == 2
        self.is_locked = state == 0

    def can_overlap(self, agent: GridAgent) -> bool:
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self, agent: GridAgent) -> bool:
        return self.is_open

    def toggle(self, env, agent: GridAgent) -> bool:
        if self.is_locked:
            if any([isinstance(obj, Key) for obj in agent.inventory]):
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self, encode_char=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            self.state = 2
        elif self.is_locked:
            self.state = 0
        # if door is closed and unlocked
        elif not self.is_open:
            self.state = 1
        else:
            raise ValueError(
                f"There is no possible state encoding for the state:\n -Door Open: {self.is_open}\n -Door Closed: {not self.is_open}\n -Door Locked: {self.is_locked}"
            )

        return super().encode(encode_char=encode_char)

    def render(self, tile_img):

        if self.state == 2:
            fill_coords(
                tile_img, point_in_rect(0.88, 1.00, 0.00, 1.00), self.color
            )
            fill_coords(
                tile_img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0)
            )
            return

        # Door frame and door
        if self.state == 0:
            fill_coords(
                tile_img, point_in_rect(0.00, 1.00, 0.00, 1.00), self.color
            )
            fill_coords(
                tile_img,
                point_in_rect(0.06, 0.94, 0.06, 0.94),
                0.45 * np.array(self.color),
            )

            # Draw key slot
            fill_coords(
                tile_img, point_in_rect(0.52, 0.75, 0.50, 0.56), self.color
            )
        else:
            fill_coords(
                tile_img, point_in_rect(0.00, 1.00, 0.00, 1.00), self.color
            )
            fill_coords(
                tile_img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0)
            )
            fill_coords(
                tile_img, point_in_rect(0.08, 0.92, 0.08, 0.92), self.color
            )
            fill_coords(
                tile_img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0)
            )

            # Draw door handle
            fill_coords(
                tile_img, point_in_circle(cx=0.75, cy=0.50, r=0.08), self.color
            )


register_object(Door.object_id, Door, scope="global")
