import numpy as np

from cogrid.constants import GridConstants


def ascii_to_numpy(ascii_list):
    rows, cols = len(ascii_list), len(ascii_list[0])
    for row in range(0, rows):
        assert len(ascii_list[row]) == cols, print("The ascii map is not rectangular!")
    arr = np.full((rows, cols), GridConstants.FreeSpace)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            arr[row, col] = ascii_list[row][col]
    return arr


def adjacent_positions(row, col):
    for rdelta, cdelta in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        yield row + rdelta, col + cdelta
