"""GridObj base class and GridAgent wrapper.

GridObj defines an object in the CoGridEnv environment. It is largely derived
from the Minigrid WorldObj:
https://github.com/Farama-Foundation/Minigrid/minigrid/core/world_object.py

GridAgent is a GridObj subclass that represents an agent on the grid,
handling direction-based rendering and inventory display.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from cogrid.constants import GridConstants
from cogrid.core.directions import Directions

if TYPE_CHECKING:
    from cogrid.rendering.tile_surface import TileSurface


class GridObj:
    """Base class for all objects that can exist on a grid cell."""

    object_id: str = None
    color: str | tuple = None
    background_color: tuple | None = None
    char: str = None

    def __init__(self, state: int = 0) -> None:
        """Initialize grid object with state."""
        self.state: int = state

        # If an object can be placed on top of this one, this will hold the object that's on top.
        self.obj_placed_on: GridObj | None = None

        self.pos: tuple[int, int] | None = None

    def encode(self, encode_char: bool = True, scope: str = "global") -> tuple[str | int, int, int]:
        """Encode this object as a (char/idx, extra, state) tuple."""
        from cogrid.core.objects.registry import object_to_idx

        return (
            self.char if encode_char else object_to_idx(self, scope=scope),
            0,
            int(self.state),
        )

    def render(self, surface: TileSurface) -> None:
        """By default, everything will be rendered as a square with the specified color."""
        surface.rect(x=0, y=0, w=1, h=1, color=self.color)

    @staticmethod
    def decode(char_or_idx: str | int, state: int, scope: str = "global") -> GridObj | None:
        """Decode a char/idx and state into a GridObj instance."""
        from cogrid.core.objects.registry import (
            get_object_id_from_char,
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
            object_id = get_object_id_from_char(char_or_idx, scope=scope)
        else:
            raise ValueError(f"Invalid identifier for decoding: {char_or_idx}")

        state = int(state)

        return make_object(object_id, state=state, scope=scope)


def _is_str(chk: Any) -> bool:
    """Check if value is a string type (including numpy str)."""
    return isinstance(chk, str) or isinstance(chk, np.str)


def _is_int(chk: Any) -> bool:
    """Check if value is an integer type (including numpy int)."""
    return isinstance(chk, int) or isinstance(chk, np.int)


class GridAgent(GridObj):
    """Grid wrapper for an Agent, handling direction rendering and inventory."""

    def __init__(self, agent: Any, n_agents: int, scope: str = "global") -> None:
        """Initialize from an Agent, encoding direction as char and inventory as state."""
        from cogrid.core.objects.registry import object_to_idx

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

    def render(self, surface: TileSurface) -> None:
        """Draw agent as a directional triangle with inventory items."""
        assert self.dir is not None
        assert len(self.inventory) <= 3, (
            "Inventory items are rendered at 1/3 size; cannot fit more than 3."
        )

        # Triangle pointing right at dir=0; rotated +90° per direction step.
        base_pts = [(0.12, 0.19), (0.87, 0.50), (0.12, 0.81)]
        theta = 0.5 * math.pi * self.dir
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotated = [
            (
                0.5 + (px - 0.5) * cos_t - (py - 0.5) * sin_t,
                0.5 + (px - 0.5) * sin_t + (py - 0.5) * cos_t,
            )
            for px, py in base_pts
        ]
        surface.polygon(points=rotated, color=self.color)

        # Inventory column down the left edge of the cell.
        inv_size = 0.30
        inv_offset = 0.04
        for i, obj in enumerate(self.inventory):
            sub = surface.subregion(
                x=inv_offset,
                y=inv_offset + i * inv_size,
                w=inv_size,
                h=inv_size,
                id_prefix=f"inv-{i}-",
            )
            obj.render(sub)

    @staticmethod
    def decode(char_or_idx: str | int, state: int, scope: str = "global") -> GridObj | None:
        """Decode a char/idx and state into a GridAgent-compatible object."""
        from cogrid.core.objects.registry import (
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

        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
