from ...graphic import *
from ...constant import *
from typing import Optional
from collections import Counter

SIGNAL_NOISE_RATIO = 0.85
REPEAT_RATIO = 1.49


def find_quadrant(x: int, y: int, w: int, h: int)->int:
    smaller_x, smaller_y = x < w, y < h
    if smaller_x and smaller_y:
        return 0
    if (not smaller_x) and smaller_y:
        return 1
    if (not smaller_x) and (not smaller_y):
        return 2
    return 3


def cal_tile(grid: Grid)->Optional[Grid]:
    for i in range(1, grid.height+1):
        for j in range(1, grid.width+1):
            if i == grid.height and j == grid.width:
                continue
            if i*j < 3:
                continue
            result = _check_tile(grid, j, i)
            if result is not None:
                return result
    return None


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
    if signal/(tile_width*tile_height) < REPEAT_RATIO:
        return None
    return result
