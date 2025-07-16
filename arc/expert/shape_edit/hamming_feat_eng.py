from ...graphic import *
from ...constant import *
import math
from typing import Optional
from scipy.stats import mode

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
    return _majority_vote([
        grid.safe_access(x-1, y), grid.safe_access(x, y-1),
        grid.safe_access(x+1, y), grid.safe_access(x, y+1)])


def diagonal(grid: Grid, x: int, y: int)->int:
    return _majority_vote([
        grid.safe_access(x-1, y-1), grid.safe_access(x+1, y-1),
        grid.safe_access(x-1, y+1), grid.safe_access(x+1, y+1)])


def mirror(grid: Grid, x: int, y: int)->int:
    neg_x, neg_y = grid.width-x-1, grid.height-y-1
    return _majority_vote([
        grid.safe_access(neg_x, y), grid.safe_access(x, neg_y),
        grid.safe_access(neg_x, neg_y)])


def get_tile(tile: Grid, x: int, y: int)->int:
    return tile.data[y % tile.height][x % tile.width]

# =======================
# private functions
# =======================


def _check_tile(grid: Grid, tile_width: int, tile_height: int)->Optional[Grid]:
    tile_data = {}
    for i in range(grid.height):
        for j in range(grid.width):
            cell = grid.data[i][j]
            if cell != NULL_COLOR:
                coord = Coordinate(j % tile_width, i % tile_height)
                tile_data[coord] = tile_data.get(coord, [])+[cell]

    signal, total = 0, 0
    tile_grid = make_grid(tile_width, tile_height)
    for coord, colors in tile_data.items():
        total += len(colors)
        mode_result = mode(colors, keepdims=False)
        signal += mode_result.count
        tile_grid.safe_assign_c(coord, mode_result.mode)

    if signal/total < SIGNAL_NOISE_RATIO:
        return None
    return tile_grid


def _majority_vote(pixels: list[int])->int:
    counts = {}
    for pixel in pixels:
        if valid_color(pixel):
            counts[pixel] = counts.get(pixel, 0)+1
    if len(counts) == 0:
        return NULL_COLOR
    if len(counts) == 1:
        return next(iter(counts.keys()))
    sorted_pairs = sorted(list(counts.items()), key=lambda x: -x[1])
    if sorted_pairs[0][1] == sorted_pairs[1][1]:
        return NULL_COLOR
    return sorted_pairs[0][0]
