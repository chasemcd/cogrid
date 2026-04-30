"""Concrete global grid object definitions.

Contains the built-in object types (Wall, Floor, Counter, Key, Door) that are
registered in the global scope via @register_object_type.
"""

from cogrid.constants import GridConstants
from cogrid.core import constants
from cogrid.core.objects.base import GridObj
from cogrid.core.objects.registry import (
    idx_to_object,
    make_object,
    register_object_type,
)
from cogrid.core.objects.when import when
from cogrid.rendering.tile_surface import TileSurface


@register_object_type("wall")
class Wall(GridObj):
    """An impassable wall tile."""

    color = constants.Colors.Grey
    char = "#"
    is_wall = True

    def __init__(self, *args, **kwargs):
        """Initialize wall with default state."""
        super().__init__(state=0)


@register_object_type("floor")
class Floor(GridObj):
    """An empty floor tile that agents can walk over."""

    color = constants.Colors.PaleBlue
    char = GridConstants.FreeSpace
    can_overlap = when()

    def __init__(self, **kwargs):
        """Initialize floor with default state."""
        super().__init__(
            state=0,
        )


@register_object_type("counter")
class Counter(GridObj):
    """A counter surface that can hold one object on top."""

    color = constants.Colors.LightBrown
    char = "C"
    can_place_on = when()
    can_pickup_from = when()

    def __init__(self, state: int = 0, **kwargs):
        """Initialize counter with given state."""
        super().__init__(
            state=state,
        )

    def render(self, surface: TileSurface) -> None:
        """Draw counter and any object placed on it."""
        super().render(surface)

        if self.obj_placed_on is not None:
            self.obj_placed_on.render(surface)

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


@register_object_type("key")
class Key(GridObj):
    """A key that can be picked up to unlock doors."""

    color = constants.Colors.Yellow
    char = "K"
    can_pickup = when()

    def __init__(self, state=0):
        """Initialize key with given state."""
        super().__init__(state=state)

    def render(self, surface: TileSurface) -> None:
        """Draw key icon with ring and teeth."""
        surface.rect(x=0.50, y=0.31, w=0.13, h=0.57, color=self.color)
        surface.rect(x=0.38, y=0.59, w=0.12, h=0.07, color=self.color)
        surface.rect(x=0.38, y=0.81, w=0.12, h=0.07, color=self.color)
        surface.circle(x=0.56, y=0.28, radius=0.190, color=self.color)
        surface.circle(x=0.56, y=0.28, radius=0.064, color=(0, 0, 0))


@register_object_type("door")
class Door(GridObj):
    """A door that can be open, closed, or locked (requires Key)."""

    color = constants.Colors.DarkGrey
    char = "D"

    def __init__(self, state):
        """Initialize door with state (0=locked, 1=closed, 2=open)."""
        super().__init__(state=state)
        self.is_open = state == 2
        self.is_locked = state == 0

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

    def render(self, surface: TileSurface) -> None:
        """Draw the door based on its state (open, closed, or locked)."""
        if self.state == 2:
            surface.rect(x=0.88, y=0.00, w=0.12, h=1.00, color=self.color)
            surface.rect(x=0.92, y=0.04, w=0.04, h=0.92, color=(0, 0, 0))
            return

        if self.state == 0:
            darker = tuple(int(c * 0.45) for c in self.color)
            surface.rect(x=0.00, y=0.00, w=1.00, h=1.00, color=self.color)
            surface.rect(x=0.06, y=0.06, w=0.88, h=0.88, color=darker)
            surface.rect(x=0.52, y=0.50, w=0.23, h=0.06, color=self.color)
        else:
            surface.rect(x=0.00, y=0.00, w=1.00, h=1.00, color=self.color)
            surface.rect(x=0.04, y=0.04, w=0.92, h=0.92, color=(0, 0, 0))
            surface.rect(x=0.08, y=0.08, w=0.84, h=0.84, color=self.color)
            surface.rect(x=0.12, y=0.12, w=0.76, h=0.76, color=(0, 0, 0))
            surface.circle(x=0.75, y=0.50, radius=0.08, color=self.color)
