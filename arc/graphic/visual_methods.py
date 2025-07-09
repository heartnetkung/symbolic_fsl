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


def split_shape(shape: Shape)->list[Shape]:
    if shape.__class__ != Unknown:
        return []

    grid = shape._grid
    divider1 = _split_shape_single_direction(grid)
    divider2 = _split_shape_single_direction(grid.flip_v())
    divider = divider1 if divider1 != -1 else divider2
    if divider != -1:
        x1, y1, grid1 = trim(np.array(grid.data)[0:divider, :])
        x2, y2, grid2 = trim(np.array(grid.data)[divider:, :])
        return [Unknown(shape.x+x1, shape.y+y1, grid1),
                Unknown(shape.x+x2, shape.y+y2+divider, grid2)]

    divider3 = _split_shape_single_direction(grid.transpose())
    divider4 = _split_shape_single_direction(grid.transpose().flip_v())
    divider = divider3 if divider3 != -1 else divider4
    if divider != -1:
        x1, y1, grid1 = trim(np.array(grid.data)[:, 0:divider])
        x2, y2, grid2 = trim(np.array(grid.data)[:, divider:])
        return [Unknown(shape.x+x1, shape.y+y1, grid1),
                Unknown(shape.x+x2+divider, shape.y+y2, grid2)]

    return []


# ====================
#  private methods
# ====================

def _split_shape_single_direction(grid: Grid)->int:
    total_height = grid.height
    left_height = 0
    for i in range(total_height):
        if grid.data[i][0] != NULL_COLOR:
            break
        left_height += 1

    right_height = 0
    for i in range(total_height):
        if grid.data[-i-1][-1] != NULL_COLOR:
            break
        right_height += 1

    if (left_height + right_height) < total_height:
        return -1
    return left_height


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
