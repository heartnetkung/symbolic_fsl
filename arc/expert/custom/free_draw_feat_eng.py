from ...graphic import *
from ...constant import *
from typing import Optional
from collections import Counter
from ..util import *
import math

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


def cal_tile(grid: Grid, check_null: bool)->Optional[Grid]:
    candidates = []
    for i in range(1, grid.height+1):
        for j in range(1, grid.width+1):
            if i == grid.height and j == grid.width:
                continue
            if i*j < 3:
                continue
            new_result = _check_tile(grid, j, i, check_null)
            if new_result is not None:
                candidates.append(new_result)
                break

    for j in range(1, grid.width):
        new_result = _check_tile(grid, j, grid.height, check_null)
        if new_result is not None:
            candidates.append(new_result)

    for i in range(1, grid.height):
        new_result = _check_tile(grid, grid.width, i, check_null)
        if new_result is not None:
            candidates.append(new_result)

    if len(candidates) == 0:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def unscale(grid: Grid)->Optional[Grid]:
    result = np.array(grid.data)
    for i in range(math.floor(grid.height/2), 1, -1):
        new_result = _check_unscale(result, i)
        if new_result is not None:
            result = new_result
            break

    result = result.transpose()
    for i in range(math.floor(grid.width/2), 1, -1):
        new_result = _check_unscale(result, i)
        if new_result is not None:
            result = new_result
            break
    result = result.transpose()

    return Grid(result.tolist()) if not np.array_equal(grid.data, result) else None


def adjacent(grid: Grid, x: int, y: int)->int:
    return vote_pixels([
        grid.safe_access(x-1, y), grid.safe_access(x, y-1),
        grid.safe_access(x+1, y), grid.safe_access(x, y+1)])


def diagonal(grid: Grid, x: int, y: int)->int:
    return vote_pixels([
        grid.safe_access(x-1, y-1), grid.safe_access(x+1, y-1),
        grid.safe_access(x-1, y+1), grid.safe_access(x+1, y+1)])


# ============== private methods ======================

def _check_tile(grid: Grid, tile_width: int, tile_height: int,
                check_null: bool)->Optional[tuple[Grid, float]]:
    tile_data = [[Counter()for _ in range(tile_width)] for _ in range(tile_height)]
    for i in range(grid.height):
        for j in range(grid.width):
            cell = grid.data[i][j]
            if (cell == NULL_COLOR) and check_null:
                continue
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

    if total == 0:
        return None

    snr = signal/total
    if snr < SIGNAL_NOISE_RATIO:
        return None
    if signal/(tile_width*tile_height) < REPEAT_RATIO:
        return None
    return result, snr


def _check_unscale(grid: np.ndarray, height: int)->Optional[np.ndarray]:
    index = np.arange(grid.shape[0])
    splitted = [grid[(index % height) == i] for i in range(height)]
    first_subgrid = splitted[0]
    for next_subgrid in splitted[1:]:
        if not np.array_equal(first_subgrid, next_subgrid):
            return None
    return first_subgrid
