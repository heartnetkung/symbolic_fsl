import numpy as np
from copy import deepcopy
from .util import *
from .shape import *
from typing import Type, Optional, Any
from ..constant import NULL_COLOR

# ==========================
# private/protect methods
# ==========================


def _expand(grid: Grid, i: int, j: int, current_obj: int,
            color: int, diagonal: bool, check_color: bool)->None:
    # grid boundary
    if i < 0 or j < 0 or i >= grid.height or j >= grid.width:
        return
    if grid.data[i][j] < 0:
        return
    if grid.data[i][j] != color and check_color:
        return
    grid.data[i][j] = current_obj
    _expand(grid, i+1, j, current_obj, color, diagonal, check_color)
    _expand(grid, i-1, j, current_obj, color, diagonal, check_color)
    _expand(grid, i, j+1, current_obj, color, diagonal, check_color)
    _expand(grid, i, j-1, current_obj, color, diagonal, check_color)
    if diagonal:
        _expand(grid, i+1, j+1, current_obj, color, diagonal, check_color)
        _expand(grid, i+1, j-1, current_obj, color, diagonal, check_color)
        _expand(grid, i-1, j+1, current_obj, color, diagonal, check_color)
        _expand(grid, i-1, j-1, current_obj, color, diagonal, check_color)


def _cal_mask_grid(grid: Grid, diagonal: bool,
                   check_color: bool)->tuple[Grid, dict[int, int]]:
    grid2 = deepcopy(grid)
    current_obj = -2
    color_map = {}

    for j in range(grid.width):
        for i in range(grid.height):
            if grid2.data[i][j] < 0:
                continue
            _expand(grid2, i, j, current_obj, grid.data[i][j], diagonal, check_color)
            color_map[current_obj] = grid.data[i][j]
            current_obj -= 1
    return grid2, color_map


def _cal_bound(mask_grid: np.ndarray)->tuple[int, int, int, int]:
    row_sum = np.sum(mask_grid != NULL_COLOR, axis=0)  # type: ignore
    col_sum = np.sum(mask_grid != NULL_COLOR, axis=1)  # type: ignore
    min_y, min_x = 0, 0
    max_x, max_y = len(row_sum), len(col_sum)
    for row_sum_i in row_sum:
        if row_sum_i != 0:
            break
        min_x += 1
    for row_sum_i in row_sum[::-1]:
        if row_sum_i != 0:
            break
        max_x -= 1
    for col_sum_j in col_sum:
        if col_sum_j != 0:
            break
        min_y += 1
    for col_sum_j in col_sum[::-1]:
        if col_sum_j != 0:
            break
        max_y -= 1
    return min_x, min_y, max_x, max_y


def trim(mask_grid: np.ndarray)->tuple[int, int, Grid]:
    min_x, min_y, max_x, max_y = _cal_bound(mask_grid)
    return min_x, min_y, Grid(mask_grid[min_y:max_y, min_x:max_x].tolist())


def _stats_mode(arr: list[Any])->Any:
    if len(arr) == 0:
        return None
    counts = {}
    for el in arr:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return sorted(counts.items(), key=lambda x: -x[1])[0][0]

# ==========================
# public methods
# ==========================


def from_grid(x: int, y: int, grid: Grid, hint: Optional[Type] = None,
              full_grid: Optional[Grid] = None)->Shape:
    if hint is not None and hint.is_valid(x, y, grid):
        if full_grid is not None:
            return hint.from_grid(x, y, grid)
        return hint.from_grid(x, y, grid)
    if FilledRectangle.is_valid(x, y, grid):
        return FilledRectangle.from_grid(x, y, grid)
    if HollowRectangle.is_valid(x, y, grid):
        return HollowRectangle.from_grid(x, y, grid)
    if full_grid is not None and Diagonal.is_valid(x, y, grid):
        return Diagonal.from_grid(x, y, grid)
    return Unknown.from_grid(x, y, grid)


def list_objects(grid: Grid, diagonal: bool = False)->list[Shape]:
    '''List all objects seen in the grid using backtrack algorithm.'''
    grid2, color_map = _cal_mask_grid(grid, diagonal, True)
    from_grid_args = []

    for current_obj in color_map.keys():
        color = color_map[current_obj]
        x, y, subgrid = trim(
            np.where(np.array(grid2.data) == current_obj, color, NULL_COLOR))
        type_ = type(from_grid(x, y, subgrid, full_grid=grid))
        from_grid_args.append((x, y, subgrid, type_))

    most_common_type = _stats_mode(list(map(lambda x: x[3], from_grid_args)))
    if most_common_type == Unknown:
        most_common_type = None  # do not force unknown on others

    return [from_grid(args[0], args[1], args[2], most_common_type, grid)
            for args in from_grid_args]


def list_sparse_objects(grid: Grid, diagonal=True)->list[Shape]:
    '''List all objects seen in the grid using backtrack algorithm.'''
    grid2, color_map = _cal_mask_grid(grid, diagonal, False)
    from_grid_args = []

    for current_obj in color_map.keys():
        color = color_map[current_obj]
        x, y, subgrid = trim(np.where(np.array(grid2.data) ==
                                      current_obj, grid.data, NULL_COLOR))
        type_ = type(from_grid(x, y, subgrid, full_grid=None))
        from_grid_args.append((x, y, subgrid, type_))

    most_common_type = _stats_mode(list(map(lambda x: x[3], from_grid_args)))
    if most_common_type == Unknown:
        most_common_type = None  # do not force unknown on others

    return [from_grid(args[0], args[1], args[2], most_common_type, None)
            for args in from_grid_args]


def list_cells(grid: Grid)->list[Shape]:
    ''''List all cells separately.'''
    result = []
    for y, row in enumerate(grid.data):
        for x, cell in enumerate(row):
            if cell != NULL_COLOR:
                result.append(FilledRectangle(x, y, 1, 1, cell))
    return result


def partition(grid: Grid, rows: list[int], cols: list[int],
              row_colors: list[int], col_colors: list[int])->list[Shape]:
    np_grid = np.array(grid.data)
    result, start_i, start_j = [], 0, 0
    assert not(len(row_colors) == 0 and len(col_colors) == 0)
    sep_color = col_colors[0] if len(row_colors) == 0 else row_colors[0]

    for i in rows:
        for j in cols:
            subgrid = np_grid[start_i:i, start_j:j]
            if subgrid.shape[0] != 0 and subgrid.shape[1] != 0:
                subgrid_obj = Grid(subgrid.tolist()).replace_color(
                    sep_color, NULL_COLOR)
                if subgrid_obj.list_colors() != {NULL_COLOR}:
                    result.append(Unknown.from_grid(start_j, start_i, subgrid_obj))
            start_j = j+1
        start_i = i+1
        start_j = 0
    return result
