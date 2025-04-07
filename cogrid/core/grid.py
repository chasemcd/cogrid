"""
Grid representation derived from Minigrid:
https://github.com/Farama-Foundation/Minigrid/minigrid/core/grid.py
"""

from __future__ import annotations

from typing import Any, Callable
from copy import deepcopy

import numpy as np

from cogrid.core.grid_object import GridObj, Wall, object_to_idx, GridAgent
from cogrid.core import grid_object
from cogrid.core.constants import CoreConstants
from cogrid.constants import GridConstants
from cogrid.visualization.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
)

CHANNEL_FIRST = False


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

    tile_cache: dict[tuple[Any], Any] = {}

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
        # TODO(chase): Add the ability to pass an encoding.
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

    def set(self, row: int, col: int, v: GridObj | None) -> None:
        """Set a GridObj at a given position in the grid.

        :param row: The row index.
        :type row: int
        :param col: The column index.
        :type col: int
        :param v: The GridObj (or None) to set.
        :type v: GridObj | None
        """
        assert (
            0 <= col < self.width
        ), f"column index {col} outside of grid of width {self.width}"
        assert (
            0 <= row < self.height
        ), f"row index {row} outside of grid of height {self.height}"
        self.grid[row * self.width + col] = v

    def get(self, row: int, col: int) -> GridObj | None:
        """Get the GridObj at a given position in the grid.

        :param row: The row index.
        :type row: int
        :param col: The column index.
        :type col: int
        :return: The GridObj at the given position, if any.
        :rtype: GridObj | None
        """
        assert (
            0 <= col < self.width
        ), f"column index {col} outside of grid of width {self.width}"
        assert (
            0 <= row < self.height
        ), f"row index {row} outside of grid of height {self.height}"
        assert self.grid is not None
        return self.grid[row * self.width + col]

    def tick(self):
        """Tick the grid, which calls the tick() method of all GridObjs.
        This is useful for any GridOjbs with time-dependent behavior.
        """
        for grid_obj in self.grid:
            if grid_obj is not None:
                grid_obj.tick()

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
            self.set(row=row, col=col + i, v=obj_type())

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
            self.set(row=row + j, col=col, v=obj_type())

    def wall_rect(
        self, col: int, row: int, w: int, h: int, grid_obj: GridObj = Wall
    ):
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

    def rotate_left(self) -> Grid:
        """Rotate the grid to the left (counter-clockwise)

        :return: The rotated grid.
        :rtype: Grid
        """
        grid = Grid(width=self.height, height=self.width)

        for col in range(self.width):
            for row in range(self.height):
                v = self.get(row=row, col=col)
                if v:
                    v.rotate_left()

                new_row = grid.height - 1 - col
                new_col = row
                grid.set(row=new_row, col=new_col, v=v)

                agent = get_grid_agent_at_position(self, (row, col))
                if agent:
                    rotated_agent = deepcopy(agent)
                    rotated_agent.rotate_left()
                    rotated_agent.pos = (new_row, new_col)
                    grid.grid_agents[rotated_agent.agent_id] = rotated_agent

        return grid

    def slice(self, topX: int, topY: int, width: int, height: int) -> Grid:
        """Get a subset of the grid

        :param topX: The top x-coordinate of the slice.
        :type topX: int
        :param topY: The top y-coordinate of the slice.
        :type topY: int
        :param width: Width of the slice.
        :type width: int
        :param height: Height of the slice.
        :type height: int
        :return: The sliced grid.
        :rtype: Grid
        """
        grid = Grid(height=height, width=width)
        for row in range(height):
            for col in range(width):
                x = topX + col
                y = topY + row
                if 0 <= x < self.width and 0 <= y < self.height:
                    v = self.get(row=y, col=x)

                    agent = get_grid_agent_at_position(
                        grid=self, position=(y, x)
                    )
                    if agent is not None:
                        grid_slice_agent = deepcopy(agent)
                        grid_slice_agent.pos = (row, col)
                        grid.grid_agents[grid_slice_agent.agent_id] = (
                            grid_slice_agent
                        )
                else:
                    v = Wall()

                grid.set(row=row, col=col, v=v)

        return grid

    def render_tile(
        self,
        obj: GridObj | None,
        highlight: bool = False,
        position: tuple[int, int] | None = None,
        tile_size: int = CoreConstants.TilePixels,
        subdivs: int = 3,
    ) -> np.ndarray:
        """Render a tile and cache the result.

        :param obj: The GridObj to render on the tile, defaults to None.
        :type obj: GridObj | None
        :param highlight: If the cell should be highlighted, defaults to False
        :type highlight: bool, optional
        :param position: Position of the GridObj (useful for retrieving GridAgent), defaults to None
        :type position: tuple[int, int] | None, optional
        :param tile_size: Rendered size of the tile, defaults to CoreConstants.TilePixels
        :type tile_size: int, optional
        :param subdivs: Number of sub-divisions of the tile, defaults to 3
        :type subdivs: int, optional
        :return: An RGB image of the rendered tile.
        :rtype: np.ndarray
        """
        grid_agent = get_grid_agent_at_position(grid=self, position=position)
        agent_dir = grid_agent.dir if grid_agent else None
        agent_color = grid_agent.agent_id if grid_agent else None
        agent_inventory_names = (
            tuple([obj.object_id for obj in grid_agent.inventory])
            if grid_agent
            else None
        )
        key: tuple[Any, ...] = (
            (agent_dir, agent_color, agent_inventory_names),
            highlight,
            tile_size,
        )
        key = obj.encode() + key if obj else key

        if key in self.__class__.tile_cache:
            return self.__class__.tile_cache[key]

        tile_img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw grid lines (separating each tile)
        fill_coords(tile_img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(tile_img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # Render the object itself onto the tile
        if obj is not None:
            obj.render(tile_img)

        if grid_agent is not None:
            grid_agent.render(tile_img)

        if highlight:
            highlight_img(tile_img)

        tile_img = downsample(tile_img, subdivs)

        # Cache the rendered tile
        self.__class__.tile_cache[key] = tile_img

        return tile_img

    def render(
        self,
        tile_size: int,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render a tile and cache the result.

        :param tile_size: The size of the tile (pixels).
        :type tile_size: int
        :param highlight_mask: Array mask to indicate which cells are highlighted, defaults to None
        :type highlight_mask: np.ndarray | None, optional
        :return: An RGB image of the rendered grid.
        :rtype: np.ndarray
        """

        # TODO(chase): Make efficiency improvements here. One option is to use
        # "dirty tiles" so we only render the tiles that have changed.

        if highlight_mask is None:
            highlight_mask = np.zeros(
                shape=(self.width, self.height), dtype=bool
            )

        # Compute total size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # render the grid
        assert highlight_mask is not None
        for row in range(self.height):
            for col in range(self.width):
                cell = self.get(row=row, col=col)

                tile_img = self.render_tile(
                    cell,
                    highlight=highlight_mask.T[row, col],
                    position=(row, col),
                    tile_size=tile_size,
                )
                ymin = row * tile_size
                ymax = (row + 1) * tile_size
                xmin = col * tile_size
                xmax = (col + 1) * tile_size

                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

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
    def decode(
        array: np.ndarray, scope: str = "global"
    ) -> tuple[Grid, np.ndarray]:
        """Decode ASCII encoding back into a Grid"""

        channels, height, width = array.shape
        assert channels == 2

        vis_mask = np.ones(shape=(height, width), dtype=bool)

        grid = Grid(height=height, width=width)
        agent_count = 0
        for col in range(width):
            for row in range(height):
                char, state = array[:, row, col]
                state = int(float(state))
                v = GridObj.decode(char, state, scope=scope)
                if v:
                    v.pos = v.init_pos = (row, col)
                    vis_mask[row, col] = v.visible()

                if isinstance(v, GridAgent):
                    grid.grid_agents[agent_count] = v
                    agent_count += 1
                else:
                    grid.set(row=row, col=col, v=v)

        return grid, vis_mask

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

    def process_vis(self, agent_pos: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        mask[agent_pos[1], agent_pos[0]] = True

        for j in reversed(range(0, self.height)):
            for i in range(0, self.width - 1):
                if not mask[i, j]:
                    continue

                cell = self.get(j, i)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, self.width)):
                if not mask[i, j]:
                    continue

                cell = self.get(j, i)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, self.height):
            for i in range(0, self.width):
                if not mask[i, j]:
                    self.set(j, i, None)

        mask = np.transpose(mask, axes=(1, 0))
        return mask
