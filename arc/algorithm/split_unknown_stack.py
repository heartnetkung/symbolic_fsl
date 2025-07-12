from ..base import *
from ..graphic import *
from copy import deepcopy

THRESHOLD = 0.6


def split_unknown_stack(shape: Shape, subshapes: list[Shape])->Optional[list[Shape]]:
    result = []
    for subshape in subshapes:
        offsets = _find_subshape_approx(shape._grid, subshape._grid)
        if offsets is not None:
            result.append(Unknown(offsets[0], offsets[1], subshape._grid))

    if len(result) < 2:
        return None
    subtract_shape = shape
    for result_shape in result:
        subtract_shape = subtract(subtract_shape, result_shape)

    if subtract_shape.mass == 0:
        return result
    return None


def _find_subshape_approx(larger: Grid, smaller: Grid)->Optional[tuple[int, int]]:
    width_diff = larger.width - smaller.width
    height_diff = larger.height - smaller.height
    max_matched = 0.01
    max_offset = (-1, -1)

    for offset_y in range(height_diff+1):
        for offset_x in range(width_diff+1):
            matched = larger.offset_subshape_approx(
                smaller, offset_x, offset_y, THRESHOLD)
            if matched > max_matched:
                max_matched = matched
                max_offset = (offset_x, offset_y)
    return None if max_matched == 0.01 else max_offset
