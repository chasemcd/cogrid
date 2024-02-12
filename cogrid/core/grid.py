"""
Grid representation derived from Minigrid:
https://github.com/Farama-Foundation/Minigrid/minigrid/core/grid.py
"""

from __future__ import annotations

from typing import Any, Callable
from copy import deepcopy

import numpy as np

from cogrid.core.grid_object import GridObj, Wall, object_to_idx, GridAgent
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
    for grid_agent in grid.grid_agents.values():
        # assert agent.pos is not None, "Agent pos should never be None."
        if np.array_equal(grid_agent.pos, position):
            return grid_agent
    return None


class Grid:
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, height: int, width: int):
        assert height >= 3 and width >= 3, "Both dimensions must be >= 3."

        self.height: int = height
        self.width: int = width

        self.grid: list[GridObj | None] = [None] * (height * width)
        self.grid_agents: dict[str:GridAgent] = {}

    def __contains__(self, item: Any) -> bool:
        assert isinstance(item, GridObj)
        if isinstance(item, GridObj):
            for grid_obj in self.grid:
                if grid_obj is item:
                    return True

        # elif isinstance(item, tuple):
        #     for e in self.grid:
        #         if e is None:
        #             continue
        #         if (e.color, e.type) == item:
        #             return True
        #         if item[0] is None and item[1] == e.type:
        #             return True
        return False

    def __eq__(self, other_grid: Grid) -> bool:
        g1 = self.encode()
        g2 = other_grid.encode()
        return np.array_equal(g1, g2)

    def __ne__(self, other: Grid) -> bool:
        return not self == other

    def copy(self) -> Grid:
        return deepcopy(self)

    def set(self, row: int, col: int, v: GridObj | None):
        assert (
            0 <= col < self.width
        ), f"column index {col} outside of grid of width {self.width}"
        assert (
            0 <= row < self.height
        ), f"row index {row} outside of grid of height {self.height}"
        self.grid[row * self.width + col] = v

    def get(self, row: int, col: int) -> GridObj | None:
        assert (
            0 <= col < self.width
        ), f"column index {col} outside of grid of width {self.width}"
        assert (
            0 <= row < self.height
        ), f"row index {row} outside of grid of height {self.height}"
        assert self.grid is not None
        return self.grid[row * self.width + col]

    def horz_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], GridObj] = Wall,
    ):
        if length is None:
            length = self.width - x
        for i in range(length):
            self.set(row=y, col=x + i, v=obj_type())

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], GridObj] = None,
    ):
        if length is None:
            length = self.height - y
        for j in range(length):
            self.set(row=y + j, col=x, v=obj_type())

    def wall_rect(self, x: int, y: int, w: int, h: int):
        self.horz_wall(x=x, y=y, length=w)
        self.horz_wall(x=x, y=y + h - 1, length=w)
        self.vert_wall(x=x, y=y, length=h)
        self.vert_wall(x=x + w - 1, y=y, length=h)

    def rotate_left(self) -> Grid:
        """Rotate the grid to the left (counter-clockwise"""
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
        """Get a subset of the grid"""
        grid = Grid(height=height, width=width)
        for row in range(height):
            for col in range(width):
                x = topX + col
                y = topY + row
                if 0 <= x < self.width and 0 <= y < self.height:
                    v = self.get(row=y, col=x)

                    agent = get_grid_agent_at_position(grid=self, position=(y, x))
                    if agent is not None:
                        grid_slice_agent = deepcopy(agent)
                        grid_slice_agent.pos = (row, col)
                        grid.grid_agents[grid_slice_agent.agent_id] = grid_slice_agent
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
        """Render a tile and cache the result."""
        grid_agent = get_grid_agent_at_position(grid=self, position=position)
        agent_dir = grid_agent.dir if grid_agent else None
        agent_color = grid_agent.color if grid_agent else None
        agent_inventory_names = (
            tuple([obj.name for obj in grid_agent.inventory]) if grid_agent else None
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
        agent_pos: tuple[int, int] | None = None,
        agent_dir: int | None = None,
    ) -> np.ndarray:
        """Render this grid at a given scale"""

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute total size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # render the grid
        assert highlight_mask is not None
        for row in range(self.height):
            for col in range(self.width):
                cell = self.get(row=row, col=col)

                # if not agent_pos and not agent_dir:
                #     agent_here = (row, col) in [tuple(agent.pos) for agent in agents.values()]
                #     agent_dir = None
                #     if agent_here:
                #         agent_dir = [agent.dir for agent in agents.values() if np.array_equal(agent.pos, (row, col))][0]
                # else:
                #     agent_here = np.array_equal(agent_pos, (row, col))

                tile_img = self.render_tile(
                    cell,
                    highlight=highlight_mask[row, col],
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
        self, vis_mask: np.ndarray | None = None, encode_char=False
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
                        GridConstants.FreeSpace if encode_char else object_to_idx(None),
                        0,
                        0,
                    )
                else:
                    encoding = v.encode(encode_char=encode_char)

                array[row, col] = encoding

        for grid_agent in self.grid_agents.values():
            row, col = grid_agent.pos
            array[row, col] = grid_agent.encode(encode_char=encode_char)

        if not encode_char:
            array = array.astype(np.int8)

        return array

    @staticmethod
    def decode(array: np.ndarray) -> tuple[Grid, np.ndarray]:
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
                v = GridObj.decode(char, state)
                if v:
                    v.pos = v.init_pos = (row, col)
                    vis_mask[row, col] = v.visible()

                if isinstance(v, GridAgent):
                    grid.grid_agents[f"agent-{agent_count}"] = v
                    agent_count += 1
                else:
                    grid.set(row=row, col=col, v=v)

        return grid, vis_mask

    # def process_vis(self, agent_pos: tuple[int, int]) -> np.ndarray:
    #     """
    #     Process the view for a single agent and mask the view behind any objects for which
    #     see_behind() is False
    #     """
    #     mask = np.zeros(shape=(self.height, self.width), dtype=bool)
    #
    #     mask[agent_pos[0], agent_pos[1]] = True
    #
    #     for row in reversed(range(0, self.height)):
    #         for col in range(0, self.width - 1):
    #             if not mask[row, col]:
    #                 continue
    #
    #             cell = self.get(row=row, col=col)
    #             if cell and not cell.see_behind():
    #                 continue
    #
    #             mask[row, col + 1] = True
    #             if row > 0:
    #                 mask[row - 1, col + 1] = True
    #                 mask[row - 1, col] = True
    #
    #         for col in reversed(range(1, self.width)):
    #             if not mask[row, col]:
    #                 continue
    #
    #             cell = self.get(row=row, col=col)
    #             if cell and not cell.see_behind():
    #                 continue
    #
    #             mask[row, col - 1] = True
    #             if row > 0:
    #                 mask[row - 1, col - 1] = True
    #                 mask[row - 1, col] = True
    #
    #     for row in range(0, self.height):
    #         for col in range(0, self.width):
    #             if not mask[row, col]:
    #                 self.set(row=row, col=col, v=None)
    #
    #     return mask

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
        # mask = np.zeros(shape=(self.height, self.width), dtype=bool)
        #
        # mask[agent_pos[0], agent_pos[1]] = True
        #
        # for j in reversed(range(0, self.height)):
        #     for i in range(0, self.width - 1):
        #         if not mask[j, i]:
        #             continue
        #
        #         cell = self.get(j, i)
        #         if cell and not cell.see_behind():
        #             continue
        #
        #         mask[j, i + 1] = True
        #         if j > 0:
        #             mask[j - 1, i + 1] = True
        #             mask[j - 1, i] = True
        #
        #     for i in reversed(range(1, self.width)):
        #         if not mask[j, i]:
        #             continue
        #
        #         cell = self.get(j, i)
        #         if cell and not cell.see_behind():
        #             continue
        #
        #         mask[j, i - 1] = True
        #         if j > 0:
        #             mask[j - 1, i - 1] = True
        #             mask[j - 1, i] = True
        #
        # for j in range(0, self.height):
        #     for i in range(0, self.width):
        #         if not mask[j, i]:
        #             self.set(j, i, None)
        #
        # return mask
