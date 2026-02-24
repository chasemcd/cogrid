"""GridObj base class and GridAgent wrapper.

GridObj defines an object in the CoGridEnv environment. It is largely derived
from the Minigrid WorldObj:
https://github.com/Farama-Foundation/Minigrid/minigrid/core/world_object.py

GridAgent is a GridObj subclass that represents an agent on the grid,
handling direction-based rendering and inventory display.
"""

from __future__ import annotations

import math
import uuid
from copy import deepcopy

import numpy as np

from cogrid.constants import GridConstants
from cogrid.core.directions import Directions
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)


class GridObj:
    """Base class for all objects that can exist on a grid cell."""

    object_id: str = None
    color: str | tuple = None
    char: str = None

    def __init__(self, state: int = 0):
        """Initialize grid object with state."""
        self.uuid: str = str(uuid.uuid4())

        self.state: int = state

        # If an object can be placed on top of this one, this will hold the object that's on top.
        self.obj_placed_on: GridObj | None = None

        # position info
        self.init_pos: tuple[int, int] | None = None
        self.pos: tuple[int, int] | None = None

    def see_behind(self, agent: GridAgent) -> bool:
        """Can the agent see through this object?"""
        return True

    def visible(self) -> bool:
        """Return True if this object is visible to agents."""
        return True

    def encode(self, encode_char=True, scope: str = "global"):
        """Encode this object as a (char/idx, extra, state) tuple."""
        from cogrid.core.grid_object_registry import object_to_idx

        return (
            self.char if encode_char else object_to_idx(self, scope=scope),
            0,
            int(self.state),
        )

    def render(self, tile_img):
        """By default, everything will be rendered as a square with the specified color."""
        fill_coords(tile_img, point_in_rect(0, 1, 0, 1), color=self.color)

    @staticmethod
    def decode(char_or_idx: str | int, state: int, scope: str = "global"):
        """Decode a char/idx and state into a GridObj instance."""
        from cogrid.core.grid_object_registry import get_object_id_from_char, make_object

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
        """Rotate this object counter-clockwise (overridden by agents)."""
        pass

    def tick(self):
        """Advance time-dependent state (overridden by objects like pots)."""
        pass


def _is_str(chk):
    """Check if value is a string type (including numpy str)."""
    return isinstance(chk, str) or isinstance(chk, np.str)


def _is_int(chk):
    """Check if value is an integer type (including numpy int)."""
    return isinstance(chk, int) or isinstance(chk, np.int)


class GridAgent(GridObj):
    """Grid wrapper for an Agent, handling direction rendering and inventory."""

    def __init__(self, agent, n_agents: int, scope: str = "global"):
        """Initialize from an Agent, encoding direction as char and inventory as state."""
        from cogrid.core.grid_object_registry import object_to_idx

        self.char = {
            Directions.Up: "^",
            Directions.Down: "v",
            Directions.Left: "<",
            Directions.Right: ">",
        }[agent.dir]

        assert len(agent.inventory) <= 1, (
            "Current implementation requires maximum inventory size of 1."
        )

        self.object_id = f"agent_{self.char}"

        state = 0 if len(agent.inventory) == 0 else object_to_idx(agent.inventory[0], scope=scope)

        super().__init__(state=state)
        self.dir = agent.dir
        self.pos = agent.pos
        self.front_pos = agent.front_pos
        self.agent_id = agent.id
        self.inventory: list[GridObj] = deepcopy(agent.inventory)
        assert self.pos is not None

        # Generate high-contrast colors based on HSV color space
        # Hue values are evenly spaced around the color wheel
        hue = (agent.agent_number - 1) * (360 / n_agents)
        # Use high saturation (0.7-1.0) and value (0.8-1.0) for vibrant colors
        # This avoids whites (high V, low S), blacks (low V), and greys (low S)
        rgb_color = self._hsv_to_rgb(hue, 0.35, 0.99)
        self.color = rgb_color

    def rotate_left(self):
        """Rotate the agent's direction counter-clockwise."""
        self.char = {"^": "<", "<": "v", "v": ">", ">": "^"}[self.char]
        self.object_id = f"agent_{self.char}"
        self.dir -= 1
        if self.dir < 0:
            self.dir += 4

    def render(self, tile_img):
        """Draw agent as a directional triangle with inventory items."""
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the triangle based on agent direction
        assert self.dir is not None
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(tile_img, tri_fn, self.color)

        # add any item in the inventory to the corner
        inv_tile_rows, inv_tile_cols = (
            tile_img.shape[0] // 3,
            tile_img.shape[1] // 3,
        )
        assert len(self.inventory) <= 3, (
            "We're rendering inventory items at 1/3 size, so can't do more than 3!"
        )

        offset = 4  # offset so we still see grid lines
        for i, obj in enumerate(self.inventory):
            inventory_tile = np.zeros(shape=(inv_tile_rows, inv_tile_cols, 3))
            obj.render(inventory_tile)

            # Take the subset of the image that we'll fill, then only
            # fill where the image is non-zero (transparent background).
            tile_subset = tile_img[
                i * inv_tile_rows + offset : (i + 1) * inv_tile_rows + offset,
                offset : inv_tile_cols + offset,
                :,
            ]
            nonzero_entries = np.nonzero(inventory_tile)
            tile_subset[nonzero_entries] = inventory_tile[nonzero_entries]

    @staticmethod
    def decode(char_or_idx: str | int, state: int, scope: str = "global"):
        """Decode a char/idx and state into a GridAgent-compatible object."""
        from cogrid.core.grid_object_registry import (
            get_object_id_from_char,
            get_object_names,
            make_object,
        )

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
            object_id = get_object_names(scope=scope)[char_or_idx]
        else:
            raise ValueError(f"Invalid identifier for decoding: {char_or_idx}")

        state = int(state)

        return make_object(object_id, state=state)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
        """Convert HSV color values to RGB tuple."""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0)
