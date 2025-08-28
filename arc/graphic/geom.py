from copy import deepcopy
from .shape import Shape, Unknown
from .util import *
import numpy as np
from .grid_methods import *
from ..constant import NULL_COLOR
from enum import Enum
from .types import Coordinate

DUMMY_VALID_COLOR = 9
N, S = Coordinate(0, -1), Coordinate(0, 1)
E, W = Coordinate(1, 0), Coordinate(-1, 0)
NE, NW = Coordinate(1, -1), Coordinate(-1, -1)
SE, SW = Coordinate(1, 1), Coordinate(-1, 1)


class LogicType(Enum):
    and_ = 0
    xor = 1
    nand = 2


def apply_logic(shapes: list[Shape], color: int, type: LogicType)->Optional[Unknown]:
    '''
    Apply logic operation to all shapes with the given color being true, false otherwise.

    Return Unknown with the size of the given grid.
    '''
    assert len(shapes) > 0

    bool_out = np.array(shapes[0]._grid.data) == color
    try:
        for shape in shapes[1:]:
            bool_current = np.array(shape._grid.data) == color
            if type == LogicType.and_:
                bool_out = np.logical_and(bool_out, bool_current)
            elif type == LogicType.xor:
                bool_out = np.logical_xor(bool_out, bool_current)
            elif type == LogicType.nand:
                bool_out = np.logical_not(np.logical_and(bool_out, bool_current))
            else:
                raise Exception('unsupported')
    except ValueError:
        return None

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


def find_outer_edge(grid: Grid)->Grid:
    output = make_grid(grid.width+2, grid.height+2)
    offset = Coordinate(-1, -1)

    # from north-south
    for x in range(grid.width+2):
        for y in range(grid.height+2):
            coord = Coordinate(x, y)+offset
            a1, a2, a3 = coord+S, coord+SE, coord+SW
            if _handle_cell(output, x, y, grid.safe_access_c(a1),
                            grid.safe_access_c(a2), grid.safe_access_c(a3)):
                break

        for y in range(grid.height+1, -1, -1):
            coord = Coordinate(x, y)+offset
            a1, a2, a3 = coord+N, coord+NE, coord+NW
            if _handle_cell(output, x, y, grid.safe_access_c(a1),
                            grid.safe_access_c(a2), grid.safe_access_c(a3)):
                break

    # from east-west
    for y in range(grid.height+2):
        for x in range(grid.width+2):
            coord = Coordinate(x, y)+offset
            a1, a2, a3 = coord+E, coord+SE, coord+NE
            if _handle_cell(output, x, y, grid.safe_access_c(a1),
                            grid.safe_access_c(a2), grid.safe_access_c(a3)):
                break

        for x in range(grid.width+1, -1, -1):
            coord = Coordinate(x, y)+offset
            a1, a2, a3 = coord+W, coord+SW, coord+NW
            if _handle_cell(output, x, y, grid.safe_access_c(a1),
                            grid.safe_access_c(a2), grid.safe_access_c(a3)):
                break

    return output


def _handle_cell(output: Grid, x: int, y: int, ahead1: int,
                 ahead2: int, ahead3: int)->bool:
    for cell_ahead in (ahead1, ahead2, ahead3):
        if cell_ahead >= 0:
            output.data[y][x] = 1
    return ahead1 >= 0
