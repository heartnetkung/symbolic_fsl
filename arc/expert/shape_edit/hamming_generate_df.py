import pandas as pd
import numpy as np
from ...base import *
from ...graphic import *
from ...ml import *
from .hamming_feat_eng import *
from ..util import *

COLS = [
    # misc
    'grid_width', 'grid_height', 'x', 'y', 'x%2', 'y%2',
    # nearby pixels
    'cell(x,y)', 'cell(x-1,y)', 'cell(x,y-1)', 'cell(x-1,y-1)', 'cell(x+1,y)',
    'cell(x,y+1)', 'cell(x+1,y+1)', 'cell(x+1,y-1)', 'cell(x-1,y+1)',
    # transformed pixels
    'cell(- x,y)', 'cell(x,- y)', 'cell(- x,-y)', 'cell(y,x)',
    # feat_eng
    'adjacent(x,y)', 'diagonal(x,y)', 'mirror(x,y)', 'tile(x,y)'
]


def generate_pixel_df(grids: list[Grid], shapes: list[Shape])->pd.DataFrame:
    result = {col: [] for col in COLS}
    for grid, shape in zip(grids, shapes):
        _gen_df(grid, shape, result)
    return pd.DataFrame(_ensure_size(result))


def _gen_df(canvas: Grid, shape: Shape, result: dict[str, list])->None:
    properties = shape.to_input_var() | _get_extra_property(shape)
    grid, canvas_width, canvas_height = shape._grid, canvas.width, canvas.height
    tile = cal_tile(grid)
    h_symmetry = find_h_symmetry(grid)
    v_symmetry = find_v_symmetry(grid)

    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.safe_access(x, y)
            if cell == NULL_COLOR:
                continue

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

            # transformed pixels
            neg_x, neg_y = x, y
            if h_symmetry is not None:
                height, offset_y = h_symmetry
                neg_y = height + (2*offset_y) - y - 1
            if v_symmetry is not None:
                width, offset_x = v_symmetry
                neg_x = width + (2*offset_x) - x - 1

            result['cell(- x,y)'].append(max(-1, grid.safe_access(neg_x, y)))
            result['cell(x,- y)'].append(max(-1, grid.safe_access(x, neg_y)))
            result['cell(- x,-y)'].append(max(-1, grid.safe_access(neg_x, neg_y)))
            if grid.width == grid.height:  # transpose requires square matrix
                result['cell(y,x)'].append(grid.safe_access(y, x))

            # feat_eng
            result['adjacent(x,y)'].append(adjacent(grid, x, y))
            result['diagonal(x,y)'].append(diagonal(grid, x, y))
            result['mirror(x,y)'].append(vote_pixels([
                result['cell(- x,y)'][-1], result['cell(x,- y)'][-1],
                result['cell(- x,-y)'][-1]]))
            if tile is not None:
                result['tile(x,y)'].append(get_tile(tile, x, y))


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
