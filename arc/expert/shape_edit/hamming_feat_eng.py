from ...graphic import *
from ...constant import *
import math
from typing import Optional
from collections import Counter
from ..util import *

SIGNAL_NOISE_RATIO = 0.85

# =======================
# public functions
# =======================


def cal_tile(grid: Grid)->Optional[Grid]:
    for i in range(2, grid.height+1):
        for j in range(2, grid.width+1):
            if i == grid.height and j == grid.width:
                continue
            result = _check_tile(grid, j, i)
            if result is not None:
                return result
    return None


def adjacent(grid: Grid, x: int, y: int)->int:
    return vote_pixels([
        grid.safe_access(x-1, y), grid.safe_access(x, y-1),
        grid.safe_access(x+1, y), grid.safe_access(x, y+1)])


def diagonal(grid: Grid, x: int, y: int)->int:
    return vote_pixels([
        grid.safe_access(x-1, y-1), grid.safe_access(x+1, y-1),
        grid.safe_access(x-1, y+1), grid.safe_access(x+1, y+1)])


def get_tile(tile: Grid, x: int, y: int)->int:
    return tile.data[y % tile.height][x % tile.width]

# =======================
# private functions
# =======================


def _check_tile(grid: Grid, tile_width: int, tile_height: int)->Optional[Grid]:
    tile_data = [[Counter()for _ in range(tile_width)] for _ in range(tile_height)]
    for i in range(grid.height):
        for j in range(grid.width):
            cell = grid.data[i][j]
            if cell != NULL_COLOR:
                tile_data[i % tile_height][j % tile_width].update([cell])

    result = make_grid(tile_width, tile_height)
    signal, total = 0, 0
    for i in range(tile_height):
        for j in range(tile_width):
            counter = tile_data[i][j]
            mode = counter.most_common(1)
            if len(mode) == 0:
                return None

            total += counter.total()
            signal += mode[0][1]
            result.data[i][j] = mode[0][0]

    if signal/total < SIGNAL_NOISE_RATIO:
        return None
    return result
