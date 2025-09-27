from .util import *
from .shape import *
from copy import deepcopy
from .grid_methods import *
from math import sqrt
from ..constant import NULL_COLOR
from .types import *


def shape_value(shape: Shape)->int:
    if shape.__class__ == Unknown:
        return 0

    masked_grid = deepcopy(shape._grid.data)
    for row in masked_grid:
        for i, cell in enumerate(row):
            row[i] = 1 if cell != NULL_COLOR else 0
    return hash(repr(masked_grid))


def colorize(shape: Shape, color: int)->Shape:
    if isinstance(shape, FilledRectangle):
        return FilledRectangle(shape.x, shape.y, shape.width, shape.height, color)
    elif isinstance(shape, HollowRectangle):
        return HollowRectangle(shape.x, shape.y, shape.width, shape.height,
                               color, shape.stroke)
    elif isinstance(shape, Diagonal):
        return Diagonal(shape.x, shape.y, shape.width, color, shape.north_west)
    elif isinstance(shape, Unknown):
        return Unknown(shape.x, shape.y, shape.grid.colorize(color))
    else:
        raise Exception('unknown shape implementation')


def list_shape_colors(shape: Shape)->set[int]:
    if shape.__class__ == Unknown:
        return shape._grid.list_colors()-{NULL_COLOR}
    return {shape.color}-{NULL_COLOR}  # type:ignore


def measure_gap(shape1: Shape, shape2: Shape, threshold: float)->float:
    center1, center2 = _find_center(shape1), _find_center(shape2)
    if _cal_upperbound_gap(center1, center2, shape1, shape2) > threshold:
        return threshold

    candidates1 = _select_candidates(shape1, center2)
    candidates2 = _select_candidates(shape2, center1)

    min_distance = threshold**2
    for candidate1 in candidates1:
        for candidate2 in candidates2:
            distance = _sq_distance(candidate1, candidate2)
            if distance < min_distance:
                min_distance = distance
    return sqrt(min_distance)


# ====================
#  private methods
# ====================

def _find_center(shape: Shape)->FloatCoordinate:
    return FloatCoordinate(shape.x+(shape.width/2), shape.y+(shape.height/2))


def _sq_distance(coord1: FloatCoordinate, coord2: FloatCoordinate)->float:
    return (coord1.x-coord2.x)**2 + (coord1.y-coord2.y)**2


def _select_candidates(shape: Shape,
                       other_center: FloatCoordinate)->list[FloatCoordinate]:
    result, grid = [], shape._grid
    for i in range(shape.height):
        for j in range(shape.width):
            if grid.data[i][j] != NULL_COLOR:
                result.append(FloatCoordinate(shape.x+j+0.5, shape.y+i+0.5))

    candidate_count = shape.width+shape.height
    sorted_result = sorted(result, key=lambda x: _sq_distance(x, other_center))
    return sorted_result[:candidate_count]


def _cal_upperbound_gap(center1: FloatCoordinate, center2: FloatCoordinate,
                        shape1: Shape, shape2: Shape)->float:
    center_distance = np.sqrt(_sq_distance(center1, center2))
    radius1 = np.sqrt(_sq_distance(FloatCoordinate(shape1.width, 0),
                                   FloatCoordinate(0, shape1.height)))
    radius2 = np.sqrt(_sq_distance(FloatCoordinate(shape2.width, 0),
                                   FloatCoordinate(0, shape2.height)))
    return center_distance-radius1-radius2
