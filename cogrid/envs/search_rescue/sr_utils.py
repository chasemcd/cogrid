import dataclasses

import numpy as np

from cogrid import constants
from cogrid.core import grid_utils


def generate_sr_grid(
    shape=None,
    load=None,
    num_green=4,
    num_yellow=3,
    num_red=2,
    num_agents=2,
    populate_rubble=True,
    np_random=None,
):
    if np_random is None:
        np_random = np.random.RandomState(seed=42)

    if load is not None:
        return getattr(FixedGrids, load)

    # First channel is characters, second channel is state
    grid = np.full((*shape, 2), fill_value=constants.GridConstants.FreeSpace)
    grid[:, :, 1] = 0

    # Fill outside border with walls
    grid[0, :] = constants.GridConstants.Wall
    grid[-1, :] = constants.GridConstants.Wall
    grid[:, 0] = constants.GridConstants.Wall
    grid[:, -1] = constants.GridConstants.Wall

    # Get free space indices
    free_spaces = list(np.argwhere(grid[:, :, 0] == constants.GridConstants.FreeSpace))

    objs_to_place = num_green + num_yellow + num_red + num_agents
    assert (
        len(free_spaces) >= objs_to_place
    ), "Not enough free spaces for specified number of objects!"

    np_random.shuffle(free_spaces)

    for _ in range(num_green):
        r, c = free_spaces.pop()
        grid[r, c, 0] = "G"

    for _ in range(num_yellow):
        r, c = free_spaces.pop()
        grid[r, c, 0] = "Y"

    for _ in range(num_red):
        r, c = free_spaces.pop()
        grid[r, c, 0] = "R"

    # add spawns
    for _ in range(num_agents):
        r, c = free_spaces.pop()
        grid[r, c, 0] = "S"

    if populate_rubble:
        # populate rubble around yellow and red
        for r, c in np.argwhere(grid == "Y"):
            grid[:, :, 0] = surround_by_rubble(grid[:, :, 0], r, c)
        for r, c in np.argwhere(grid == "R"):
            grid[:, :, 0] = surround_by_rubble(grid[:, :, 0], r, c)

    return grid


def surround_by_rubble(grid, row, col):
    for r, c in grid_utils.adjacent_positions(row, col):
        if grid[r, c] == constants.GridConstants.FreeSpace:
            grid[r, c] = constants.GridConstants.Rubble
    return grid


@dataclasses.dataclass
class FixedGrids:
    m3minimap = [
        "#############",
        "#S  S#      #",
        "#  X #      #",
        "# XR #XYXG  #",
        "#    # X  XY#",
        "#   X#     X#",
        "#  XY#      #",
        "#    #  RX  #",
        "##X###  X   #",
        "#    #     X#",
        "#    #G   XR#",
        "#  X #     X#",
        "# XYX#    # #",
        "#    ###X# ##",
        "#G   #    G #",
        "#    X     G#",
        "#    #X     #",
        "#    #YXX # #",
        "# X  #X RX ##",
        "# RX #  #####",
        "#   #########",
        "#############",
    ]

    m3minimap_reduced = [
        "#############",
        "#S  S       #",
        "#  X   X    #",
        "# XRX XYXG  #",
        "#  XG  X  XY#",
        "#  XYX      #",
        "#   X     XX#",
        "#XX       XX#",
        "#RX  X    XR#",
        "#############",
    ]
