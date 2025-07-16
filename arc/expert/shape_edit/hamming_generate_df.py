import pandas as pd
import numpy as np
from ...base import *
from ...graphic import *
from ...ml import *
from .hamming_feat_eng import *

COLS = {
    # misc
    'grid_width', 'grid_height', 'x', 'y',
    # nearby pixels
    'cell(x-1,y)', 'cell(x,y-1)', 'cell(x-1,y-1)', 'cell(x+1,y)', 'cell(x,y+1)',
    'cell(x+1,y+1)', 'cell(x+1,y-1)', 'cell(x-1,y+1)', 'cell(x,y)',
    # transformed pixels
    'cell(-x,y)', 'cell(x,-y)', 'cell(-x,-y)',
    'cell(y,x)', 'cell(y,-x)', 'cell(-y,x)', 'cell(-y,-x)',
    # feat_eng
    'adjacent(x,y)', 'diagonal(x,y)', 'mirror(x,y)', 'tile(x,y)'
}


def generate_pixel_df(grids: list[Grid], shapes: list[Shape])->pd.DataFrame:
    result = {col: [] for col in COLS}
    for grid, shape in zip(grids, shapes):
        _gen_df(grid, shape, result)
    return pd.DataFrame(_ensure_size(result))


def _gen_df(canvas: Grid, shape: Shape, result: dict[str, list])->None:
    properties = shape.to_input_var() | _get_extra_property(shape)
    grid, canvas_width, canvas_height = shape._grid, canvas.width, canvas.height
    tile = cal_tile(grid)

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
