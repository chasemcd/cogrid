"""Concrete global grid object definitions.

Contains the built-in object types (Wall, Floor, Counter, Key, Door) that are
registered in the global scope via @register_object_type.
"""

import numpy as np

from cogrid.constants import GridConstants
from cogrid.core import constants
from cogrid.core.grid_object_base import GridAgent, GridObj
from cogrid.core.grid_object_registry import (
    idx_to_object,
    make_object,
    register_object_type,
)
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
)


@register_object_type("wall", is_wall=True)
class Wall(GridObj):
    """An impassable wall tile."""

    object_id = "wall"
    color = constants.Colors.Grey
    char = "#"

    def __init__(self, *args, **kwargs):
        """Initialize wall with default state."""
        super().__init__(state=0)

    def see_behind(self) -> bool:
        """Return False; walls block visibility."""
        return False


@register_object_type("floor", can_overlap=True)
class Floor(GridObj):
    """An empty floor tile that agents can walk over."""

    object_id = "floor"
    color = constants.Colors.PaleBlue
    char = GridConstants.FreeSpace

    def __init__(self, **kwargs):
        """Initialize floor with default state."""
        super().__init__(
            state=0,
        )

    def can_overlap(self) -> bool:
        """Return True; agents can walk over floor tiles."""
        return True


@register_object_type("counter", can_place_on=True)
class Counter(GridObj):
    """A counter surface that can hold one object on top."""

    object_id = "counter"
    color = constants.Colors.LightBrown
    char = "C"

    def __init__(self, state: int = 0, **kwargs):
        """Initialize counter with given state."""
        super().__init__(
            state=state,
        )

    def can_place_on(self, agent: GridAgent, cell: GridObj) -> bool:
        """True when the counter has no object on it."""
        return self.obj_placed_on is None

    def render(self, tile_img):
        """Draw counter and any object placed on it."""
        super().render(tile_img)

        if self.obj_placed_on is not None:
            self.obj_placed_on.render(tile_img)

    @classmethod
    def build_render_sync_fn(cls):
        """Return a render-sync callback for counter objects."""

        def counter_render_sync(grid, env_state, scope):
            """Sync obj_placed_on for counters from object_state_map."""
            osm = env_state.object_state_map
            for r in range(grid.height):
                for c in range(grid.width):
                    cell = grid.get(r, c)
                    if cell is None or cell.object_id != "counter":
                        continue
                    state_val = int(osm[r, c])
                    if state_val > 0:
                        placed_id = idx_to_object(state_val, scope=scope)
                        cell.obj_placed_on = (
                            make_object(placed_id, scope=scope) if placed_id else None
                        )
                    else:
                        cell.obj_placed_on = None

        return counter_render_sync


@register_object_type("key", can_pickup=True)
class Key(GridObj):
    """A key that can be picked up to unlock doors."""

    object_id = "key"
    color = constants.Colors.Yellow
    char = "K"

    def __init__(self, state=0):
        """Initialize key with given state."""
        super().__init__(state=state)

    def can_pickup(self, agent: GridAgent):
        """Return True; keys are always pickable."""
        return True

    def render(self, tile_img):
        """Draw key icon with ring and teeth."""
        # Vertical quad
        fill_coords(tile_img, point_in_rect(0.50, 0.63, 0.31, 0.88), self.color)

        # Teeth
        fill_coords(tile_img, point_in_rect(0.38, 0.50, 0.59, 0.66), self.color)
        fill_coords(tile_img, point_in_rect(0.38, 0.50, 0.81, 0.88), self.color)

        # Ring
        fill_coords(tile_img, point_in_circle(cx=0.56, cy=0.28, r=0.190), self.color)
        fill_coords(tile_img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


@register_object_type("door")
class Door(GridObj):
    """A door that can be open, closed, or locked (requires Key)."""

    object_id = "door"
    color = constants.Colors.DarkGrey
    char = "D"

    def __init__(self, state):
        """Initialize door with state (0=locked, 1=closed, 2=open)."""
        super().__init__(state=state)
        self.is_open = state == 2
        self.is_locked = state == 0

    def can_overlap(self, agent: GridAgent) -> bool:
        """The agent can only walk over this cell when the door is open."""
        return self.is_open

    def see_behind(self, agent: GridAgent) -> bool:
        """Return True only when the door is open."""
        return self.is_open

    def toggle(self, env, agent: GridAgent) -> bool:
        """Unlock (if agent has Key) or open/close the door."""
        if self.is_locked:
            if any([isinstance(obj, Key) for obj in agent.inventory]):
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self, encode_char=False):
        """Encode the door as a 3-tuple of integers."""
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
                f"No possible state encoding for door: open={self.is_open}, locked={self.is_locked}"
            )

        return super().encode(encode_char=encode_char)

    def render(self, tile_img):
        """Draw the door based on its state (open, closed, or locked)."""
        if self.state == 2:
            fill_coords(tile_img, point_in_rect(0.88, 1.00, 0.00, 1.00), self.color)
            fill_coords(tile_img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.state == 0:
            fill_coords(tile_img, point_in_rect(0.00, 1.00, 0.00, 1.00), self.color)
            fill_coords(
                tile_img,
                point_in_rect(0.06, 0.94, 0.06, 0.94),
                0.45 * np.array(self.color),
            )

            # Draw key slot
            fill_coords(tile_img, point_in_rect(0.52, 0.75, 0.50, 0.56), self.color)
        else:
            fill_coords(tile_img, point_in_rect(0.00, 1.00, 0.00, 1.00), self.color)
            fill_coords(tile_img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(tile_img, point_in_rect(0.08, 0.92, 0.08, 0.92), self.color)
            fill_coords(tile_img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.50, r=0.08), self.color)
