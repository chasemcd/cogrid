"""Search-and-rescue grid object types (items, obstacles, victims)."""

from cogrid.core import constants
from cogrid.core import objects as grid_object
from cogrid.core.objects.registry import register_object_type
from cogrid.core.objects.when import when
from cogrid.rendering.tile_surface import TileSurface


@register_object_type("medkit", scope="search_rescue")
class MedKit(grid_object.GridObj):
    """A medical kit that enables rescuing yellow victims."""

    color = constants.Colors.LightPink
    char = "M"
    can_pickup = when()

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, surface: TileSurface) -> None:
        """Draw a red cross icon."""
        surface.rect(x=0.1, y=0.1, w=0.8, h=0.8, color=(255, 0, 0))
        surface.rect(x=0.4, y=0.2, w=0.2, h=0.6, color=(255, 255, 255))
        surface.rect(x=0.2, y=0.4, w=0.6, h=0.2, color=(255, 255, 255))


@register_object_type("pickaxe", scope="search_rescue")
class Pickaxe(grid_object.GridObj):
    """A tool that enables clearing rubble obstacles."""

    color = constants.Colors.Grey
    char = "T"
    can_pickup = when()

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, surface: TileSurface) -> None:
        """Draw a pickaxe with brown handle and grey head."""
        surface.rect(x=0.45, y=0.15, w=0.10, h=0.75, color=constants.Colors.Brown)
        surface.polygon(
            points=[(0.5, 0.1), (0.5, 0.3), (0.9, 0.35)],
            color=self.color,
        )
        surface.polygon(
            points=[(0.5, 0.1), (0.5, 0.3), (0.1, 0.35)],
            color=self.color,
        )


@register_object_type("rubble", scope="search_rescue")
class Rubble(grid_object.GridObj):
    """An obstacle that can be cleared by an Engineer or agent with Pickaxe."""

    color = constants.Colors.Brown
    char = "X"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, surface: TileSurface) -> None:
        """Draw three brown circles representing rubble pile."""
        surface.circle(x=0.25, y=0.3, radius=0.2, color=self.color)
        surface.circle(x=0.75, y=0.3, radius=0.2, color=self.color)
        surface.circle(x=0.50, y=0.7, radius=0.2, color=self.color)


@register_object_type("green_victim", scope="search_rescue")
class GreenVictim(grid_object.GridObj):
    """A victim rescuable by any adjacent agent."""

    color = constants.Colors.Green
    char = "G"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, surface: TileSurface) -> None:
        """Draw a green circle."""
        surface.circle(x=0.5, y=0.47, radius=0.4, color=self.color)


@register_object_type("purple_victim", scope="search_rescue")
class PurpleVictim(grid_object.GridObj):
    """A victim rescuable by any adjacent agent (higher reward)."""

    color = constants.Colors.Purple
    char = "P"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, surface: TileSurface) -> None:
        """Draw a purple circle."""
        surface.circle(x=0.5, y=0.47, radius=0.4, color=self.color)


@register_object_type("yellow_victim", scope="search_rescue")
class YellowVictim(grid_object.GridObj):
    """A victim rescuable only by a Medic or agent carrying a MedKit."""

    color = constants.Colors.Yellow
    char = "Y"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, surface: TileSurface) -> None:
        """Draw a yellow circle."""
        surface.circle(x=0.5, y=0.47, radius=0.4, color=self.color)


@register_object_type("red_victim", scope="search_rescue")
class RedVictim(grid_object.GridObj):
    """A victim requiring two-agent cooperative rescue within a time window."""

    color = constants.Colors.Red
    char = "R"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(state=state)

    def render(self, surface: TileSurface) -> None:
        """Draw a red circle."""
        surface.circle(x=0.5, y=0.47, radius=0.4, color=self.color)
