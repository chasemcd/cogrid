"""Search-and-rescue grid object types (items, obstacles, victims)."""

from cogrid.core import constants, grid_object
from cogrid.core.grid_object_registry import register_object_type
from cogrid.core.when import when
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
)


@register_object_type("medkit", scope="search_rescue")
class MedKit(grid_object.GridObj):
    """A medical kit that enables rescuing yellow victims."""

    color = constants.Colors.LightPink
    char = "M"
    can_pickup = when()

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, tile_img):
        """Draw a red cross icon."""
        # red background with white cross
        fill_coords(tile_img, point_in_rect(0.1, 0.9, 0.1, 0.9), (255, 0, 0))
        fill_coords(tile_img, point_in_rect(0.4, 0.6, 0.2, 0.8), (255, 255, 255))
        fill_coords(tile_img, point_in_rect(0.2, 0.8, 0.4, 0.6), (255, 255, 255))


@register_object_type("pickaxe", scope="search_rescue")
class Pickaxe(grid_object.GridObj):
    """A tool that enables clearing rubble obstacles."""

    color = constants.Colors.Grey
    char = "T"
    can_pickup = when()

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, tile_img):
        """Draw a pickaxe with brown handle and grey head."""
        # Brown Handle
        fill_coords(tile_img, point_in_rect(0.45, 0.55, 0.15, 0.9), constants.Colors.Brown)

        # Use two triangles to make the pickaxe head
        # These are of the specified color
        tri_fn = point_in_triangle(
            (0.5, 0.1),
            (0.5, 0.3),
            (0.9, 0.35),
        )
        fill_coords(tile_img, tri_fn, self.color)

        tri_fn = point_in_triangle(
            (0.5, 0.1),
            (0.5, 0.3),
            (0.1, 0.35),
        )
        fill_coords(tile_img, tri_fn, self.color)


@register_object_type("rubble", scope="search_rescue")
class Rubble(grid_object.GridObj):
    """An obstacle that can be cleared by an Engineer or agent with Pickaxe."""

    color = constants.Colors.Brown
    char = "X"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def see_behind(self) -> bool:
        """Return False; rubble blocks visibility."""
        return False

    def render(self, tile_img):
        """Draw three brown circles representing rubble pile."""
        fill_coords(tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), self.color)
        fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), self.color)
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), self.color)


@register_object_type("green_victim", scope="search_rescue")
class GreenVictim(grid_object.GridObj):
    """A victim rescuable by any adjacent agent."""

    color = constants.Colors.Green
    char = "G"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, tile_img):
        """Draw a green circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), self.color)


@register_object_type("purple_victim", scope="search_rescue")
class PurpleVictim(grid_object.GridObj):
    """A victim rescuable by any adjacent agent (higher reward)."""

    color = constants.Colors.Purple
    char = "P"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, tile_img):
        """Draw a purple circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), self.color)


@register_object_type("yellow_victim", scope="search_rescue")
class YellowVictim(grid_object.GridObj):
    """A victim rescuable only by a Medic or agent carrying a MedKit."""

    color = constants.Colors.Yellow
    char = "Y"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, tile_img):
        """Draw a yellow circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), self.color)


@register_object_type("red_victim", scope="search_rescue")
class RedVictim(grid_object.GridObj):
    """A victim requiring two-agent cooperative rescue within a time window."""

    color = constants.Colors.Red
    char = "R"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, tile_img):
        """Draw a red circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), self.color)
