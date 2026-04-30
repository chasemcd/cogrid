"""Grid representation derived from Minigrid.

See https://github.com/Farama-Foundation/Minigrid/minigrid/core/grid.py
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from cogrid.constants import GridConstants
from cogrid.core.constants import CoreConstants
from cogrid.core.objects import GridAgent, GridObj, Wall, object_to_idx

if TYPE_CHECKING:
    from mug.rendering import Surface

CHANNEL_FIRST = False

_GRID_LINE_COLOR = (100, 100, 100)
_GRID_LINE_FRAC = 0.031


def get_grid_agent_at_position(
    grid: Grid, position: tuple[int, int] | np.ndarray
) -> GridAgent | None:
    """Return the GridAgent at a given position, if any.

    :param grid: The grid to search.
    :type grid: Grid
    :param position: The position to search for.
    :type position: tuple[int, int] | np.ndarray
    :return: The GridAgent at the given position, if any.
    :rtype: GridAgent | None
    """
    for grid_agent in grid.grid_agents.values():
        # assert agent.pos is not None, "Agent pos should never be None."
        if np.array_equal(grid_agent.pos, position):
            return grid_agent
    return None


class Grid:
    """The Grid class is a 2D HxW grid of GridObjs.

    :param height: Height of the grid.
    :type height: int
    :param width: Width of the grid.
    :type width: int
    """

    def __init__(self, height: int, width: int):
        """Generator method for Grid class."""
        assert height >= 3 and width >= 3, "Both dimensions must be >= 3."

        self.height: int = height
        self.width: int = width

        self.grid: list[GridObj | None] = [None] * (height * width)
        self.grid_agents: dict[str, GridAgent] = {}

    def __contains__(self, item: Any) -> bool:
        """Check if an item is in the grid.

        :param item: The item to check for.
        :type item: Any
        :return: Whether the item is in the grid.
        :rtype: bool
        """
        if isinstance(item, GridObj):
            for grid_obj in self.grid:
                if grid_obj is item:
                    return True
        return False

    def __eq__(self, other_grid: Grid) -> bool:
        """Check if two grids are equal.

        :param other_grid: The other grid to compare.
        :type other_grid: Grid
        :return: Whether the two grids are equal.
        :rtype: bool
        """
        g1 = self.encode()
        g2 = other_grid.encode()
        return np.array_equal(g1, g2)

    def __ne__(self, other: Grid) -> bool:
        """Check if two grids are not equal.

        :param other: The other grid to compare.
        :type other: Grid
        :return: Whether the two grids are not equal.
        :rtype: bool
        """
        return not self == other

    def copy(self) -> Grid:
        """Return a deep copy of the grid.

        :return: A deep copy of the grid.
        :rtype: Grid
        """
        return deepcopy(self)

    def set(self, row: int, col: int, obj: GridObj | None) -> None:
        """Set a GridObj at a given position in the grid.

        :param row: The row index.
        :type row: int
        :param col: The column index.
        :type col: int
        :param obj: The GridObj (or None) to set.
        :type obj: GridObj | None
        """
        assert 0 <= col < self.width, f"column index {col} outside of grid of width {self.width}"
        assert 0 <= row < self.height, f"row index {row} outside of grid of height {self.height}"
        self.grid[row * self.width + col] = obj

    def get(self, row: int, col: int) -> GridObj | None:
        """Get the GridObj at a given position in the grid.

        :param row: The row index.
        :type row: int
        :param col: The column index.
        :type col: int
        :return: The GridObj at the given position, if any.
        :rtype: GridObj | None
        """
        assert 0 <= col < self.width, f"column index {col} outside of grid of width {self.width}"
        assert 0 <= row < self.height, f"row index {row} outside of grid of height {self.height}"
        assert self.grid is not None
        return self.grid[row * self.width + col]

    def horz_wall(
        self,
        col: int,
        row: int,
        length: int | None = None,
        obj_type: GridObj = Wall,
    ) -> None:
        """Create a horizontal wall of obj_type in the grid.

        :param col: The column index.
        :type col: int
        :param row: The row index.
        :type row: int
        :param length: The length of the wall, defaults to maximum.
        :type length: int | None, optional
        :param obj_type: GridObj to construct a wall from, defaults to Wall.
        :type obj_type: Callable[[], GridObj], optional
        """
        if length is None:
            length = self.width - col
        for i in range(length):
            self.set(row=row, col=col + i, obj=obj_type())

    def vert_wall(
        self,
        col: int,
        row: int,
        length: int | None = None,
        obj_type: GridObj = Wall,
    ) -> None:
        """Create a vertical wall of obj_type in the grid.

        :param col: The column index.
        :type col: int
        :param row: The row index.
        :type row: int
        :param length: The length of the wall, defaults to maximum.
        :type length: int | None, optional
        :param obj_type: GridObj to construct a wall from, defaults to Wall
        :type obj_type: Callable[[], GridObj], optional
        """
        if length is None:
            length = self.height - row
        for j in range(length):
            self.set(row=row + j, col=col, obj=obj_type())

    def wall_rect(self, col: int, row: int, w: int, h: int, grid_obj: GridObj = Wall):
        """Create a rectangle of walls in the grid.

        :param col: Initial column index.
        :type col: int
        :param row: Initial row index.
        :type row: int
        :param w: Width of the rectangle.
        :type w: int
        :param h: Height of the rectangle.
        :type h: int
        :param grid_obj: GridObj to construct a wall from, defaults to Wall.
        :type grid_obj: GridObj, optional
        """
        self.horz_wall(row=row, col=col, length=w, obj_type=grid_obj)
        self.horz_wall(row=row, col=col + h - 1, length=w, obj_type=grid_obj)
        self.vert_wall(row=row, col=col, length=h, obj_type=grid_obj)
        self.vert_wall(row=row + w - 1, col=col, length=h, obj_type=grid_obj)

    def draw_to_surface(
        self,
        surface: Surface,
        tile_size: int = CoreConstants.TilePixels,
        *,
        x_offset: float = 0,
        y_offset: float = 0,
    ) -> None:
        """Draw the grid onto a ``mug.Surface`` in canvas pixel space.

        Each cell is drawn into a ``TileSurface`` view that exposes 0-1
        cell-local coordinates to ``GridObj.render``. Grid lines are drawn
        per-cell (top and left edges) so cells share borders. Agents are
        drawn after their cell's object.

        :param surface: Frame-wide ``mug.Surface`` to draw onto.
        :param tile_size: Pixel size of each grid cell.
        :param x_offset: Canvas-pixel x of the grid's top-left corner.
        :param y_offset: Canvas-pixel y of the grid's top-left corner.
        """
        from cogrid.rendering.tile_surface import TileSurface

        for row in range(self.height):
            for col in range(self.width):
                ts = TileSurface(
                    surface,
                    x_offset=col * tile_size + x_offset,
                    y_offset=row * tile_size + y_offset,
                    width=tile_size,
                    height=tile_size,
                    id_prefix=f"{row}-{col}-",
                )
                ts.rect(x=0, y=0, w=_GRID_LINE_FRAC, h=1, color=_GRID_LINE_COLOR)
                ts.rect(x=0, y=0, w=1, h=_GRID_LINE_FRAC, color=_GRID_LINE_COLOR)

                cell = self.get(row=row, col=col)
                if cell is not None and cell.background_color is not None:
                    ts.rect(x=0, y=0, w=1, h=1, color=cell.background_color)
                if cell is not None:
                    cell.render(ts)

                grid_agent = get_grid_agent_at_position(grid=self, position=(row, col))
                if grid_agent is not None:
                    grid_agent.render(ts)

    def render(
        self,
        tile_size: int,
    ) -> np.ndarray:
        """Render the full grid to an RGB ``np.ndarray``.

        Builds a fresh ``mug.Surface`` and a ``PygameRenderer`` for each
        call. For repeated rendering or for MUG mode, hold a long-lived
        ``Surface`` and call :meth:`draw_to_surface` directly.
        """
        from mug.rendering import Surface

        from cogrid.rendering.raster import PygameRenderer

        width_px = self.width * tile_size
        height_px = self.height * tile_size

        surface = Surface(width=width_px, height=height_px)
        self.draw_to_surface(surface, tile_size=tile_size)

        renderer = PygameRenderer(width=width_px, height=height_px)
        renderer.apply(surface.commit().to_dict())
        return renderer.to_array()

    def encode(
        self,
        vis_mask: np.ndarray | None = None,
        encode_char=False,
        scope: str = "global",
    ) -> np.ndarray:
        """Produce an ASCII/int representation of the grid."""
        if vis_mask is None:
            vis_mask = np.ones((self.height, self.width), dtype=bool)

        array = np.empty((self.height, self.width, 3), dtype=object)

        assert vis_mask is not None
        for col in range(self.width):
            for row in range(self.height):
                if not vis_mask[row, col]:
                    continue

                v = self.get(row=row, col=col)
                if v is None:
                    encoding = (
                        (
                            GridConstants.FreeSpace
                            if encode_char
                            else object_to_idx(None, scope=scope)
                        ),
                        0,
                        0,
                    )
                else:
                    encoding = v.encode(encode_char=encode_char, scope=scope)

                array[row, col] = encoding

        for grid_agent in self.grid_agents.values():
            row, col = grid_agent.pos
            array[row, col] = grid_agent.encode(encode_char=encode_char)

        if not encode_char:
            array = array.astype(np.int8)

        return array

    @staticmethod
    def decode(array: np.ndarray, scope: str = "global") -> Grid:
        """Decode ASCII encoding back into a Grid."""
        channels, height, width = array.shape
        assert channels == 2

        grid = Grid(height=height, width=width)
        agent_count = 0
        for col in range(width):
            for row in range(height):
                char, state = array[:, row, col]
                state = int(float(state))
                v = GridObj.decode(char, state, scope=scope)
                if v:
                    v.pos = (row, col)

                if isinstance(v, GridAgent):
                    grid.grid_agents[agent_count] = v
                    agent_count += 1
                else:
                    grid.set(row=row, col=col, obj=v)

        return grid

    def get_obj_count(self, grid_obj: GridObj | None) -> int:
        """Get the number of a particular object that exists in the grid.

        :param grid_obj: The GridObj to count.
        :type grid_obj: GridObj | None
        :return: The number of the GridObj in the grid.
        :rtype: int
        """
        count = 0
        for obj in self.grid:
            if isinstance(obj, grid_obj):
                count += 1
        return count
