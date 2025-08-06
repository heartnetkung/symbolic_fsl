import pandas as pd
import numpy as np
from ...base import *
from ...graphic import *
from ...ml import *
from ..util import *
import math
from .free_draw_feat_eng import *

COLS = [
    # misc
    'grid_width', 'grid_height', 'x', 'y', 'x%2', 'y%2',
    # nearby pixels
    'cell(x,y)', 'cell(x-1,y)', 'cell(x,y-1)', 'cell(x-1,y-1)', 'cell(x+1,y)',
    'cell(x,y+1)', 'cell(x+1,y+1)', 'cell(x+1,y-1)', 'cell(x-1,y+1)',
    # basic feat eng
    f'cell(+x%w,y%h)', 'cell(+x/w,y/h)', 'tile(x,y)', 'tile2(x,y)', 'adjacent(x,y)',
    'diagonal(x,y)', 'y_mid', 'x_mid',
    'left_cell(y)', 'right_cell(y)', 'top_cell(x)', 'bottom_cell(x)',
    # advanced feat eng
    'quadrant_ccw_rotate(+x,y)', 'quadrant_cw_rotate(+x,y)', 'unscaled_cell(+x,y)'
]


def generate_pixel_df(grids: list[Grid], all_shapes: list[list[Shape]],
                      widths: list[int], heights: list[int])->pd.DataFrame:
    result = {col: [] for col in COLS}
    for grid, shapes, w, h in zip(grids, all_shapes, widths, heights):
        _gen_df(grid, shapes[0], w, h, result)
    return pd.DataFrame(_ensure_size(result))


def _gen_df(canvas: Grid, shape: Shape, w: int, h: int, result: dict[str, list])->None:
    properties = shape.to_input_var() | _get_extra_property(shape)
    grid, canvas_width, canvas_height = shape._grid, canvas.width, canvas.height
    np_grid = np.array(grid.data)
    rotations = [Grid(np.rot90(np_grid, i).tolist()) for i in range(4)]
    tile, tile2, unscaled = cal_tile(grid, True), cal_tile(grid, False), unscale(grid)

    for y in range(h):
        for x in range(w):
            cell = grid.safe_access(x, y)
            gw, gh = grid.width, grid.height
            quadrant = find_quadrant(x, y, gw, gh)

            # shape properties
            _gen_shape_properties(properties, result)

            # misc
            result['grid_width'].append(canvas_width)
            result['grid_height'].append(canvas_height)
            result['x'].append(x)
            result['y'].append(y)
            result['x%2'].append(x % 2)
            result['y%2'].append(y % 2)

            # nearby pixels
            result['cell(x-1,y)'].append(grid.safe_access(x-1, y))
            result['cell(x,y-1)'].append(grid.safe_access(x, y-1))
            result['cell(x-1,y-1)'].append(grid.safe_access(x-1, y-1))
            result['cell(x+1,y)'].append(grid.safe_access(x+1, y))
            result['cell(x,y+1)'].append(grid.safe_access(x, y+1))
            result['cell(x+1,y+1)'].append(grid.safe_access(x+1, y+1))
            result['cell(x+1,y-1)'].append(grid.safe_access(x+1, y-1))
            result['cell(x-1,y+1)'].append(grid.safe_access(x-1, y+1))
            result['cell(x,y)'].append(grid.safe_access(x, y))

            # basic feat eng
            result[f'cell(+x%w,y%h)'].append(grid.safe_access(x % gw, y % gh))
            result['cell(+x/w,y/h)'].append(grid.safe_access(
                math.floor(x/gw), math.floor(y/gh)))
            result['diagonal(x,y)'].append(diagonal(grid, x, y))
            result['adjacent(x,y)'].append(adjacent(grid, x, y))
            result['y_mid'].append(y - math.ceil(h/2))
            result['x_mid'].append(x - math.ceil(w/2))
            result['left_cell(y)'].append(grid.safe_access(0, y))
            result['right_cell(y)'].append(grid.safe_access(w-1, y))
            result['top_cell(x)'].append(grid.safe_access(x, 0))
            result['bottom_cell(x)'].append(grid.safe_access(x, h-1))
            if tile is not None:
                result['tile(x,y)'].append(tile.safe_access(
                    x % tile.width, y % tile.height))
            if tile2 is not None:
                result['tile2(x,y)'].append(tile2.safe_access(
                    x % tile2.width, y % tile2.height))

            # advanced feat eng
            result['quadrant_ccw_rotate(+x,y)'].append(
                rotations[quadrant].safe_access(x % gw, y % gh))
            result['quadrant_cw_rotate(+x,y)'].append(
                rotations[(-quadrant) % 4].safe_access(x % gw, y % gh))
            if unscaled is not None:
                result['unscaled_cell(+x,y)'].append(unscaled.safe_access(
                    x % unscaled.width, y % unscaled.height))


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
