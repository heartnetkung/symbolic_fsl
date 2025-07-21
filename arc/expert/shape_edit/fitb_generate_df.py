import pandas as pd
import numpy as np
from ...base import *
from ...graphic import *
from ...ml import *
from .fitb_feat_eng import *

COLS = {
    # misc
    'grid_width', 'grid_height', 'x', 'y',
    # nearby pixels
    'cell(x-1,y)', 'cell(x,y-1)', 'cell(x-1,y-1)', 'cell(x+1,y)', 'cell(x,y+1)',
    'cell(x+1,y+1)', 'cell(x+1,y-1)', 'cell(x-1,y+1)',
    # transformed pixels
    'cell(-x,y)', 'cell(x,-y)', 'cell(-x,-y)',
    'cell(y,x)', 'cell(y,-x)', 'cell(-y,x)', 'cell(-y,-x)',
    # feat_eng
    'adjacent(x,y)', 'diagonal(x,y)', 'is_plus_path(x,y)', 'is_cross_path(x,y)',
    'mirror(x,y)', 'double_mirror(x,y)', 'tile(x,y)', 'plus(x,y)', 'cross(x,y)',
    # global_feat_eng
    'is_leftside(x,y)', 'is_rightside(x,y)', 'is_topside(x,y)', 'is_bottomside(x,y)',
    'is_outside(x,y)', 'row_blank_count_rank(x,y)', 'col_blank_count_rank(x,y)'
}


def generate_pixel_df(grids: list[Grid], shapes: list[Shape],
                      bounds: list[tuple[int, int, int, int]])->pd.DataFrame:
    result = {col: [] for col in COLS}
    for grid, shape, bound in zip(grids, shapes, bounds):
        _gen_df(grid, shape, result, bound)
    return pd.DataFrame(_ensure_size(result))


def _gen_df(canvas: Grid, shape: Shape, result: dict[str, list],
            bound: tuple[int, int, int, int])->None:
    properties = shape.to_input_var() | _get_extra_property(shape)
    grid, canvas_width, canvas_height = shape._grid, canvas.width, canvas.height
    leftside_pixels = cal_leftside_pixels(grid)
    rightside_pixels = cal_rightside_pixels(grid)
    topside_pixels = cal_topside_pixels(grid)
    bottomside_pixels = cal_bottomside_pixels(grid)
    outside_pixels = cal_outside_pixels(grid)
    row_blank_count_rank = cal_row_blank_count_rank(grid)
    col_blank_count_rank = cal_col_blank_count_rank(grid)
    plus_color, cross_color = cal_plus(grid), cal_cross(grid)
    tile = cal_tile(grid, bound)

    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.safe_access(x, y)
            if cell != NULL_COLOR:
                continue

            # shape properties
            _gen_shape_properties(properties, result)

            # misc
            result['grid_width'].append(canvas_width)
            result['grid_height'].append(canvas_height)
            result['x'].append(x)
            result['y'].append(y)

            # nearby pixels
            result['cell(x-1,y)'].append(grid.safe_access(x-1, y))
            result['cell(x,y-1)'].append(grid.safe_access(x, y-1))
            result['cell(x-1,y-1)'].append(grid.safe_access(x-1, y-1))
            result['cell(x+1,y)'].append(grid.safe_access(x+1, y))
            result['cell(x,y+1)'].append(grid.safe_access(x, y+1))
            result['cell(x+1,y+1)'].append(grid.safe_access(x+1, y+1))
            result['cell(x+1,y-1)'].append(grid.safe_access(x+1, y-1))
            result['cell(x-1,y+1)'].append(grid.safe_access(x-1, y+1))

            # transformed pixels
            neg_x, neg_y = grid.width-x-1, grid.height-y-1
            result['cell(-x,y)'].append(grid.safe_access(neg_x, y))
            result['cell(x,-y)'].append(grid.safe_access(x, neg_y))
            result['cell(-x,-y)'].append(grid.safe_access(neg_x, neg_y))
            if grid.width == grid.height:  # transpose requires square matrix
                result['cell(y,x)'].append(grid.safe_access(y, x))
                result['cell(y,-x)'].append(grid.safe_access(y, neg_x))
                result['cell(-y,x)'].append(grid.safe_access(neg_y, x))
                result['cell(-y,-x)'].append(grid.safe_access(neg_y, neg_x))

            # feat_eng
            result['adjacent(x,y)'].append(adjacent(grid, x, y))
            result['diagonal(x,y)'].append(diagonal(grid, x, y))
            result['mirror(x,y)'].append(mirror(grid, x, y))
            result['double_mirror(x,y)'].append(double_mirror(grid, x, y))
            if grid.width == grid.height:
                success = is_cross_path(grid, x, y)
                result['is_cross_path(x,y)'].append(success)
                result['cross(x,y)'].append(cross_color if success else NULL_COLOR)
            if (grid.width % 2 == 1) and (grid.height % 2 == 1):
                success = is_plus_path(grid, x, y)
                result['is_plus_path(x,y)'].append(success)
                result['plus(x,y)'].append(plus_color if success else NULL_COLOR)
            if tile is not None:
                result['tile(x,y)'].append(get_tile(tile, x, y, bound))

            # global feat_eng
            coord = Coordinate(x, y)
            result['is_leftside(x,y)'].append(1 if coord in leftside_pixels else 0)
            result['is_rightside(x,y)'].append(1 if coord in rightside_pixels else 0)
            result['is_topside(x,y)'].append(1 if coord in topside_pixels else 0)
            result['is_bottomside(x,y)'].append(1 if coord in bottomside_pixels else 0)
            result['is_outside(x,y)'].append(1 if coord in outside_pixels else 0)
            result['row_blank_count_rank(x,y)'].append(row_blank_count_rank[y])
            result['col_blank_count_rank(x,y)'].append(col_blank_count_rank[x])


def _gen_shape_properties(properties: dict[str, int], df_data: dict[str, list])->None:
    for k, v in properties.items():
        key = f'shape.{k}'
        df_data[key] = df_data.get(key, [])+[v]


def _get_extra_property(shape: Shape)->dict[str, int]:
    result = {}
    result['least_color'] = shape._grid.get_least_color()
    return result


def _ensure_size(df_data: dict[str, list])->dict[str, list]:
    max_size = max([len(v) for v in df_data.values()])
    return {k: v for k, v in df_data.items() if len(v) == max_size}
