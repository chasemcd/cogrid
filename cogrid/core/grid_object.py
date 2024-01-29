"""GridObj defines an object in the gridworld environment. It is largely derived from the Minigrid WorldObj:
https://github.com/Farama-Foundation/Minigrid/minigrid/core/world_object.py

"""
from __future__ import annotations
from copy import deepcopy
import math

import numpy as np

from envs.gridworld.constants import GridConstants
from envs.gridworld.core.roles import Roles
from envs.gridworld.core.constants import ObjectColors, COLORS, COLOR_NAMES
from envs.gridworld.core.grid_utils import adjacent_positions
from envs.gridworld.core.directions import Directions
from envs.gridworld.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

OBJECT_NAMES = [
    None,
    "free_space",
    "wall",
    "door",
    "key",
    "pickaxe",
    "counter",
    "medkit",
    "rubble",
    "green_victim",
    "yellow_victim",
    "red_victim",
] + [f"agent_{direction}" for direction in "^>v<"]

OBJECT_NAMES_TO_CHAR = {
    "wall": GridConstants.Wall,
    "key": GridConstants.Key,
    "pickaxe": GridConstants.Pickaxe,
    "counter": GridConstants.Counter,
    "medkit": GridConstants.MedKit,
    "rubble": GridConstants.Rubble,
    "green_victim": GridConstants.GreenVictim,
    "yellow_victim": GridConstants.YellowVictim,
    "red_victim": GridConstants.RedVictim,
    "agent_^": GridConstants.AgentUp,
    "agent_>": GridConstants.AgentRight,
    "agent_v": GridConstants.AgentDown,
    "agent_<": GridConstants.AgentLeft,
    "door": GridConstants.Door,
}

OBJECT_CHAR_TO_NAMES = dict((v, k) for k, v in OBJECT_NAMES_TO_CHAR.items())


def fetch_object_by_name(name):
    if name == "agent":
        raise ValueError(
            "Agents must be added to a grid from the Gridworld env and need to be instantiated with the `Agent` object."
        )
    elif name == "wall":
        return Wall
    elif name == "floor":
        return Floor
    elif name == "counter":
        return Counter
    elif name == "key":
        return Key
    elif name == "green_victim":
        return GreenVictim
    elif name == "yellow_victim":
        return YellowVictim
    elif name == "red_victim":
        return RedVictim
    elif name == "rubble":
        return Rubble
    elif name == "medkit":
        return MedKit
    elif name == "pickaxe":
        return Pickaxe
    elif name == "door":
        return Door
    else:
        raise NotImplementedError(f"Object {name} not implemented.")


class GridObj:
    def __init__(
        self,
        name: str,
        color: tuple,
        char: str,
        state: int = 0,
        toggle_value: float = 0,
        inventory_value: float = 0,
        overlap_value: float = 0,
    ):
        self.name: str = name
        self.color: str = color
        self.char: str = char
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

    def can_overlap(self) -> bool:
        """Can an agent overlap with this object?"""
        return False

    def can_pickup(self) -> bool:
        """Can an agent pick this object up and store in inventory?"""
        return False

    def can_place_on(self) -> bool:
        """
        Can another object be placed on top of this object? e.g., a countertop that can't be walked through
        but can have an item on top of it.
        """
        return False

    def can_pickup_from(self) -> bool:
        """Can the agent pick up an object from this one?"""
        return self.obj_placed_on is not None and self.obj_placed_on.can_pickup()

    def place_on(self, cell: GridObj) -> None:
        self.obj_placed_on = cell
        self.state = object_to_idx(cell)

    def pick_up_from(self) -> GridObj:
        assert self.obj_placed_on is not None, f"Picking up from but there's no object placed on {self.name}"
        cell = self.obj_placed_on
        self.obj_placed_on = None
        self.state = 0
        return cell

    def see_behind(self) -> bool:
        """Can the agent see through this object?"""
        return True

    def visible(self) -> bool:
        return True

    def toggle(self, env, toggling_agent=None) -> bool:
        """
        Trigger/Toggle an action this object performs. Some toggles are conditioned on the environment
        and require specific conditions to be met, which can be checked with the end.
        """
        return False

    def encode(self, encode_char=True):
        return self.char if encode_char else object_to_idx(self), COLOR_NAMES.index(self.color), int(self.state)

    def render(self, tile_img):
        """By default, everything will be rendered as a square with the specified color."""
        fill_coords(tile_img, point_in_rect(0, 1, 0, 1), color=COLORS[self.color])

        if self.obj_placed_on is not None:
            self.obj_placed_on.render(tile_img)

    @staticmethod
    def decode(char_or_idx: str | int, state: int):

        if char_or_idx in [None, GridConstants.FreeSpace, GridConstants.Obscured]:
            return None

        # check if the name was passed instead of the character
        if _is_str(char_or_idx) and len(char_or_idx) > 1:
            obj_name = char_or_idx
        elif _is_str(char_or_idx):
            obj_name = OBJECT_CHAR_TO_NAMES[char_or_idx]
        elif _is_int(char_or_idx):
            obj_name = OBJECT_NAMES[char_or_idx]
        else:
            raise ValueError(f"Invalid identifier for decoding: {char_or_idx}")

        state = int(state)

        return fetch_object_by_name(obj_name)(state)

    def rotate_left(self):
        """Some objects (e.g., agents) have a rotation and must be rotated with the grid."""
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
        obj_name = object.name
    else:
        obj_name = object

    return OBJECT_NAMES.index(obj_name)


class GridAgent(GridObj):
    def __init__(self, agent):
        """
        Grid agents are initialized slightly differently. State corresponds to the object they are holding
        and char/colors are unique for each agent.
        """
        color = {
            1: ObjectColors.AgentOne,
            2: ObjectColors.AgentTwo,
            3: ObjectColors.AgentThree,
            4: ObjectColors.AgentFour,
        }[agent.agent_number]

        char = {Directions.Up: "^", Directions.Down: "v", Directions.Left: "<", Directions.Right: ">"}[agent.dir]
        assert len(agent.inventory) <= 1, "Current implementation requires maximum inventory size of 1."
        # TODO(chase): State must encapsulate carried objects and role
        state = 0 if len(agent.inventory) == 0 else object_to_idx(agent.inventory[0])
        # state = agent.role_idx
        super().__init__(name=f"agent_{char}", color=color, char=char, state=state)
        self.dir = agent.dir
        self.pos = agent.pos
        self.agent_id = agent.id
        self.inventory = deepcopy(agent.inventory)
        assert self.pos is not None

    def rotate_left(self):
        self.char = {"^": "<", "<": "v", "v": ">", ">": "^"}[self.char]
        self.name = f"agent_{self.char}"
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
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(tile_img, tri_fn, COLORS[self.color])

        # add any item in the inventory to the corner
        inv_tile_rows, inv_tile_cols = tile_img.shape[0] // 3, tile_img.shape[1] // 3
        assert len(self.inventory) <= 3, "We're rending inv items at 1/3 size, so can't do more than 3!"
        offset = 4  # offset so we still see grid lines
        for i, obj in enumerate(self.inventory):
            inventory_tile = np.zeros(shape=(inv_tile_rows, inv_tile_cols, 3))
            obj.render(inventory_tile)

            # Take the subset of the image that we'll fill, then only fill where the image is non-zero
            # this makes a transparent background, rather than adding in a black square
            tile_subset = tile_img[
                i * inv_tile_rows + offset : (i + 1) * inv_tile_rows + offset, offset : inv_tile_cols + offset, :
            ]
            nonzero_entries = np.nonzero(inventory_tile)
            tile_subset[nonzero_entries] = inventory_tile[nonzero_entries]

    @staticmethod
    def decode(char_or_idx: str | int, state: int):

        if char_or_idx in [None, GridConstants.FreeSpace, GridConstants.Obscured]:
            return None

        # check if the name was passed instead of the character
        if _is_str(char_or_idx) and len(char_or_idx) > 1:
            obj_name = char_or_idx
        elif _is_str(char_or_idx):
            obj_name = OBJECT_CHAR_TO_NAMES[char_or_idx]
        elif _is_int(char_or_idx):
            obj_name = OBJECT_NAMES[char_or_idx]
        else:
            raise ValueError(f"Invalid identifier for decoding: {char_or_idx}")

        state = int(state)

        return fetch_object_by_name(obj_name)(state)


class Wall(GridObj):
    def __init__(self, state=0):
        super().__init__(name="wall", color=ObjectColors.Wall, char=GridConstants.Wall, state=state)

    def see_behind(self) -> bool:
        return False


class Floor(GridObj):
    def __init__(self, state=0, **kwargs):
        super().__init__(name="floor", color=ObjectColors.Floor, char=GridConstants.FreeSpace, state=state)

    def can_overlap(self) -> bool:
        return True


class Counter(GridObj):
    def __init__(self, state=0, **kwargs):
        super().__init__(name="counter", color=ObjectColors.Counter, char=GridConstants.Counter, state=state)

    def can_place_on(self) -> bool:
        return self.state == 0


class Key(GridObj):
    def __init__(self, state=0):
        super().__init__(name="key", color=ObjectColors.Key, char=GridConstants.Key, state=state)

    def can_pickup(self):
        return True

    def render(self, tile_img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(tile_img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(tile_img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(tile_img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(tile_img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(tile_img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Door(GridObj):
    def __init__(self, state):
        super().__init__("door", color=ObjectColors.Door, char=GridConstants.Door, state=state)
        self.is_open = state == 2
        self.is_locked = state == 0

    def can_overlap(self) -> bool:
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self) -> bool:
        return self.is_open

    def toggle(self, env, toggling_agent=None) -> bool:
        if self.is_locked:
            if any([isinstance(obj, Key) for obj in toggling_agent.inventory]):
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
        c = COLORS[self.color]

        if self.state == 2:
            fill_coords(tile_img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(tile_img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.state == 0:
            fill_coords(tile_img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(tile_img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(tile_img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(tile_img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(tile_img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(tile_img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(tile_img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class MedKit(GridObj):
    def __init__(self, state=0):
        super().__init__(
            name="medkit",
            color=ObjectColors.MedKit,
            char=GridConstants.MedKit,
            state=state,
        )

    def can_pickup(self):
        return True

    def render(self, tile_img):
        # red background with white cross
        fill_coords(tile_img, point_in_rect(0.1, 0.9, 0.1, 0.9), (255, 0, 0))  # red background
        fill_coords(tile_img, point_in_rect(0.4, 0.6, 0.2, 0.8), (255, 255, 255))  # vertical bar
        fill_coords(tile_img, point_in_rect(0.2, 0.8, 0.4, 0.6), (255, 255, 255))  # horizontal bar


class Pickaxe(GridObj):
    def __init__(self, state=0):
        super().__init__(
            name="pickaxe",
            color=ObjectColors.Pickaxe,
            char=GridConstants.Pickaxe,
            state=state,
        )

    def can_pickup(self):
        return True

    def render(self, tile_img):
        # red background with white cross
        fill_coords(tile_img, point_in_rect(0.45, 0.55, 0.15, 0.9), COLORS["brown"])  # brown handle

        # TODO(chase): figure out the triangle rendering.
        tri_fn = point_in_triangle(
            (0.5, 0.1),
            (0.5, 0.3),
            (0.9, 0.35),
        )
        # tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5)
        fill_coords(tile_img, tri_fn, COLORS["grey"])

        tri_fn = point_in_triangle(
            (0.5, 0.1),
            (0.5, 0.3),
            (0.1, 0.35),
        )
        # tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5)
        fill_coords(tile_img, tri_fn, COLORS["grey"])

        # triangle pointing right
        # tri_fn = point_in_triangle(
        #     (0.9, 0.75),
        #     (0.5, 0.8),
        #     (0.5, 0.7),
        # )
        # tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5)
        # fill_coords(tile_img, tri_fn, COLORS["grey"])

        # # triangle pointing left
        # tri_fn = point_in_triangle(
        #     (0.5, 0.8),
        #     (0.5, 0.7),
        #     (0.1, 0.75),
        # )
        # tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * 2)
        # fill_coords(tile_img, tri_fn, COLORS["grey"])
        #


class Rubble(GridObj):
    def __init__(self, state=0):
        super().__init__(
            name="rubble", color=ObjectColors.Rubble, char=GridConstants.Rubble, state=state, toggle_value=0.05
        )

    def see_behind(self) -> bool:
        return False

    def toggle(self, env, toggling_agent=None) -> bool:
        """Rubble can be toggled by an Engineer/agent with Pickaxe"""
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        toggling_agent_is_engineer = (
            any([isinstance(obj, Pickaxe) for obj in toggling_agent.inventory]) or toggling_agent.role == Roles.Engineer
        )

        assert toggling_agent_is_adjacent, "Rubble toggled by non-adjacent agent."

        toggle_success = toggling_agent_is_engineer

        if toggle_success:
            self._remove_from_grid(env.grid)
        return toggle_success

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), c)
        fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), c)
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), c)


class GreenVictim(GridObj):
    def __init__(self, state=0):
        super().__init__(
            name="green_victim",
            color=ObjectColors.GreenVictim,
            char=GridConstants.GreenVictim,
            state=state,
            toggle_value=0.1,
        )

    def toggle(self, env, toggling_agent=None) -> bool:
        """Toggling a victim rescues them. A GreenVictim can be rescued if any agent is adjacent to it"""
        # Toggle should only be triggered if the GreenVictim is directly in front of it. For debugging purposes,
        # we'll just check to make sure that's true (if this isn't triggered it can be removed).
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        assert toggling_agent_is_adjacent, "GreenVictim toggled by non-adjacent agent."

        self._remove_from_grid(env.grid)
        return toggling_agent_is_adjacent

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


class YellowVictim(GridObj):
    def __init__(self, state=0):
        super().__init__(
            name="yellow_victim",
            color=ObjectColors.YellowVictim,
            char=GridConstants.YellowVictim,
            state=state,
            toggle_value=0.2,
        )

    def toggle(self, env, toggling_agent=None) -> bool:
        """
        Toggling a victim rescues them. A YellowVictim can be rescued if a Medic is adjcent to it or the agent
        is carrying a MedKit
        """
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        toggling_agent_is_medic = (
            any([isinstance(obj, MedKit) for obj in toggling_agent.inventory]) or toggling_agent.role == Roles.Medic
        )

        assert toggling_agent_is_adjacent, "YellowVictim toggled by non-adjacent agent."

        toggle_success = toggling_agent_is_medic

        if toggle_success:
            self._remove_from_grid(env.grid)
        return toggle_success

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


class RedVictim(GridObj):
    def __init__(self, state=0):
        super().__init__(
            name="red_victim", color=ObjectColors.RedVictim, char=GridConstants.RedVictim, state=state, toggle_value=0.3
        )

    def toggle(self, env, toggling_agent=None) -> bool:
        """A RedVictim can be rescued if a Medic (or agent carrying MedKit) is the adjacent toggling agent
        and there is another agent adjacent."""
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        toggling_agent_is_medic = (
            any([isinstance(obj, MedKit) for obj in toggling_agent.inventory]) or toggling_agent.role == Roles.Medic
        )

        assert toggling_agent_is_adjacent, "RedVictim toggled by non-adjacent agent."

        other_adjacent_agent = None
        for agent in env.agents.values():
            if agent is toggling_agent or tuple(agent.pos) not in adj_positions:
                continue
            other_adjacent_agent = agent
            break

        toggle_success = toggling_agent_is_medic and other_adjacent_agent is not None

        if toggle_success:
            self._remove_from_grid(env.grid)

            # If we're using common reward both will automatically receive. If not, both acted in saving,
            # so we reward both for the red victim.
            if not env.common_reward:
                other_adjacent_agent.reward += self.toggle_value

        return toggle_success

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)
