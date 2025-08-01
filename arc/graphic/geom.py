from copy import deepcopy
from .shape import Shape, Unknown
from .util import *
import numpy as np
from .grid_methods import *
from ..constant import NULL_COLOR
from enum import Enum

DUMMY_VALID_COLOR = 9


class LogicType(Enum):
    and_ = 0
    xor = 1
    # or does not exist because it's the "and" of the inverse


def apply_logic(shape1: Shape, shape2: Shape, color: int, type: LogicType)->Unknown:
    '''
    Apply logic operation to all shapes with the given color being true, false otherwise.

    Return Unknown with the size of the given grid.
    '''
    width = max(shape1.width, shape2.width)
    height = max(shape1.height, shape2.height)
    canvas1 = _normalize_draw(shape1, width, height)
    canvas2 = _normalize_draw(shape2, width, height)
    bool1 = np.array(canvas1.data) == color
    bool2 = np.array(canvas2.data) == color
    if type == LogicType.and_:
        bool_out = np.logical_and(bool1, bool2)
    elif type == LogicType.xor:
        bool_out = np.logical_xor(bool1, bool2)
    else:
        raise Exception('unsupported')

    # since NULL_COLOR in our algorithm denotes False value,
    # the true input value must be replaced to a dummy
    out_color = color if color is not NULL_COLOR else DUMMY_VALID_COLOR
    result = np.where(bool_out, out_color, NULL_COLOR)
    return Unknown(0, 0, Grid(result.tolist()))


def union(shape1: Shape, shape2: Shape)->Unknown:
    width = max(shape1.x+shape1.width, shape2.x+shape2.width)
    height = max(shape1.y+shape1.height, shape2.y+shape2.height)
    canvas = make_grid(width, height)
    shape1.draw(canvas)
    shape2.draw(canvas)
    x, y, grid = trim(np.array(canvas.data))
    return Unknown(x, y, grid)


def subtract(shape1: Shape, shape2: Shape)->Unknown:
    grid = deepcopy(shape1._grid)
    shape2_grid = shape2._grid.data
    for i in range(shape2.height):
        for j in range(shape2.width):
            if shape2_grid[i][j] != NULL_COLOR:
                grid.safe_assign(j+shape2.x-shape1.x, i+shape2.y-shape1.y, NULL_COLOR)
    return Unknown(shape1.x, shape1.y, grid)


def _from_full_grid(grid: np.ndarray, hint: Optional[Type] = None)->Unknown:
    x, y, altered_grid = trim(grid)
    if altered_grid.height == 0 or altered_grid.width == 0:
        return NULL_SHAPE  # intersect can generate zero with object
    return Unknown(x, y, altered_grid)


def _normalize_draw(shape: Shape, width: int, height: int)->Grid:
    canvas = make_grid(width, height)
    shape.draw(canvas, include_xy=False)
    return canvas
