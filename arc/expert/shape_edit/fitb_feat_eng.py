from ...graphic import *
from ...constant import *
from ...ml import to_rank
import math
from typing import Optional
from collections import Counter
from scipy.stats import mode

# =======================
# public functions
# =======================


def cal_leftside_pixels(grid: Grid)->set[Coordinate]:
    result = set()
    for y, row in enumerate(grid.data):
        first_index = _find_first_valid_index(row)
        result |= {Coordinate(x, y) for x in range(first_index)}
    return result


def cal_rightside_pixels(grid: Grid)->set[Coordinate]:
    result = set()
    for y, row in enumerate(grid.data):
        last_index = _find_first_valid_index(_copy_reverse(row))
        result |= {Coordinate(x, y) for x in range(grid.width-last_index, grid.width)}
    return result


def cal_topside_pixels(grid: Grid)->set[Coordinate]:
    transposed_result = cal_leftside_pixels(grid.transpose())
    return {Coordinate(coord.y, coord.x) for coord in transposed_result}


def cal_bottomside_pixels(grid: Grid)->set[Coordinate]:
    transposed_result = cal_rightside_pixels(grid.transpose())
    return {Coordinate(coord.y, coord.x) for coord in transposed_result}


def cal_outside_pixels(grid: Grid)->set[Coordinate]:
    result = set()
    for i in range(grid.height):
        _cal_outside_recursive(grid, i, 0, result)
        _cal_outside_recursive(grid, i, grid.width-1, result)
    for j in range(grid.width):
        _cal_outside_recursive(grid, 0, j, result)
        _cal_outside_recursive(grid, grid.height-1, j, result)
    return result


def cal_row_blank_count_rank(grid: Grid)->list[int]:
    return to_rank([Counter(row)[NULL_COLOR] for row in grid.data])


def cal_col_blank_count_rank(grid: Grid)->list[int]:
    return cal_row_blank_count_rank(grid.transpose())


def cal_tile(grid: Grid, bound: tuple[int, int, int, int],
             check_null: bool)->Optional[Grid]:
    min_x, max_x, min_y, max_y = bound
    original_grid = grid.crop(min_x, min_y, max_x-min_x, max_y-min_y)
    for i in range(2, original_grid.height+1):
        for j in range(2, original_grid.width+1):
            if i == original_grid.height and j == original_grid.width:
                continue
            result = _check_tile(original_grid, j, i, check_null)
            if result is not None:
                return result
    return None


def cal_plus(grid: Grid)->int:
    w, h = grid.width, grid.height
    if (w % 2 == 0) or (h % 2 == 0):
        return NULL_COLOR

    middle_x, middle_y = math.floor(grid.width/2), math.floor(grid.height/2)
    horizontal = [grid.data[middle_y][i] for i in range(w)]
    vertical = [grid.data[i][middle_x] for i in range(h)]
    return _vote_pixels(horizontal+vertical)


def cal_cross(grid: Grid)->int:
    if grid.width != grid.height:
        return NULL_COLOR

    top_left = [grid.data[i][i] for i in range(grid.height)]
    top_right = [grid.data[i][i] for i in range(grid.height-1, -1, -1)]
    return _vote_pixels(top_left+top_right)


def is_plus_path(grid: Grid, x: int, y: int)->int:
    if x == math.floor(grid.width/2):
        return 1
    if y == math.floor(grid.height/2):
        return 1
    return 0


def is_cross_path(grid: Grid, x: int, y: int)->int:
    if x == y:
        return 1
    if x+y+1 == grid.width:
        return 1
    return 0


def adjacent(grid: Grid, x: int, y: int)->int:
    return _merge_pixels([
        grid.safe_access(x-1, y), grid.safe_access(x, y-1),
        grid.safe_access(x+1, y), grid.safe_access(x, y+1)])


def diagonal(grid: Grid, x: int, y: int)->int:
    return _merge_pixels([
        grid.safe_access(x-1, y-1), grid.safe_access(x+1, y-1),
        grid.safe_access(x-1, y+1), grid.safe_access(x+1, y+1)])


def mirror(grid: Grid, x: int, y: int)->int:
    neg_x, neg_y = grid.width-x-1, grid.height-y-1
    return _merge_pixels([
        grid.safe_access(neg_x, y), grid.safe_access(x, neg_y),
        grid.safe_access(neg_x, neg_y)])


def double_mirror(grid: Grid, x: int, y: int)->int:
    neg_x, neg_y = grid.width-x-1, grid.height-y-1
    return _merge_pixels([
        grid.safe_access(neg_x, y), grid.safe_access(x, neg_y),
        grid.safe_access(neg_x, neg_y), grid.safe_access(y, x),
        grid.safe_access(neg_y, x), grid.safe_access(y, neg_x),
        grid.safe_access(neg_y, neg_x)])


def get_tile(tile: Grid, x: int, y: int, bound: tuple[int, int, int, int])->int:
    min_x, max_x, min_y, max_y = bound
    offset_x = (tile.width - (min_x % tile.width)) % tile.width
    offset_y = (tile.height - (min_y % tile.height)) % tile.height
    return tile.data[(y+offset_y) % tile.height][(x+offset_x) % tile.width]


# =======================
# private functions
# =======================


def _check_tile(grid: Grid, tile_width: int, tile_height: int,
                check_null: bool)->Optional[Grid]:
    tile_grid = make_grid(tile_width, tile_height)
    for i in range(grid.height):
        for j in range(grid.width):
            tile_i, tile_j = i % tile_height, j % tile_width
            tile_cell = tile_grid.data[tile_i][tile_j]
            cell = grid.data[i][j]
            if (cell == NULL_COLOR) and check_null:
                continue
            if tile_cell == NULL_COLOR:
                tile_grid.data[tile_i][tile_j] = cell
            elif tile_cell != cell:
                return None
    return tile_grid


def _cal_outside_recursive(grid: Grid, i: int, j: int, result: set[Coordinate])->None:
    if i < 0 or j < 0 or i >= grid.height or j >= grid.width:
        return
    coord = Coordinate(j, i)
    if coord in result:
        return
    cell = grid.data[i][j]
    if cell != NULL_COLOR:
        return
    result.add(coord)
    _cal_outside_recursive(grid, i+1, j, result)
    _cal_outside_recursive(grid, i-1, j, result)
    _cal_outside_recursive(grid, i, j+1, result)
    _cal_outside_recursive(grid, i, j-1, result)


def _find_first_valid_index(row: list[int])->int:
    for i, cell in enumerate(row):
        if valid_color(cell):
            return i
    return -1


def _copy_reverse(row: list[int])->list[int]:
    result = row.copy()
    result.reverse()
    return result


def _merge_pixels(pixels: list[int])->int:
    result = NULL_COLOR
    for pixel in pixels:
        if not valid_color(pixel):
            continue
        if result == NULL_COLOR:  # first encounter
            result = pixel
        elif result != pixel:  # multiple colors are unacceptable
            return MISSING_VALUE
    return result


def _vote_pixels(pixels: list[int])->int:
    result = []
    for pixel in pixels:
        if not valid_color(pixel):
            continue
        result.append(pixel)
    if len(result) == 0:
        return NULL_COLOR
    return mode(result).mode
