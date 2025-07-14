from ...base import *
from ...graphic import *

APPROX_THRESHOLD = 0.6
NEARBY_THRESHOLD = 1


def find_subshape(smaller: Shape, larger: Shape)->dict[str, int]:
    if (larger.width < smaller.width or larger.height < smaller.height):
        return {}

    smaller_mass, larger_mass = smaller.mass, larger.mass
    if smaller_mass >= larger_mass:
        return {}

    result = {}
    if larger._grid.find_subgrid_approx(smaller._grid, APPROX_THRESHOLD) is not None:
        result['approx_subshape'] = 1

    enable_colorless = len(smaller._grid.list_colors()-{NULL_COLOR}) == 1
    additional_result = _find_subshape(smaller._grid, larger._grid)
    if len(additional_result) > 0:
        if enable_colorless:
            if 'subshape' in additional_result:
                additional_result['colorless_subshape'] = 1
            if 'transformed_subshape' in additional_result:
                additional_result['colorless_transformed_subshape'] = 1
        return result | additional_result

    if not enable_colorless:
        return {}

    colorless_smaller_grid = smaller._grid.normalize_color()
    colorless_larger_grid = larger._grid.normalize_color()
    additional_result2 = _find_subshape(colorless_smaller_grid, colorless_larger_grid)
    if 'subshape' in additional_result2:
        additional_result2['colorless_subshape'] = 1
        del additional_result2['subshape']
    if 'transformed_subshape' in additional_result2:
        additional_result2['colorless_transformed_subshape'] = 1
        del additional_result2['transformed_subshape']

    return result | additional_result2


def _find_subshape(smaller: Grid, larger: Grid)->dict[str, int]:
    if larger.find_subgrid(smaller) is not None:
        return {'subshape': 1, 'transformed_subshape': 1, 'approx_subshape': 1}
    result = {}
    for transformed_grid in geom_transform_all(
            smaller, exclude_original=True, include_inverse=True):
        if larger.find_subgrid(transformed_grid) is not None:
            return result | {'transformed_subshape': 1}
    return result


def is_nearby(a: Shape, b: Shape)->dict[str, int]:
    x_a_range, x_b_range = range(a.x, a.x+a.width), range(b.x, b.x+b.width)
    if not _is_range_nearby(x_a_range, x_b_range):
        return {}
    y_a_range, y_b_range = range(a.y, a.y+a.height), range(b.y, b.y+b.height)
    if not _is_range_nearby(y_a_range, y_b_range):
        return {}
    return {'nearby': 1}


def _is_range_nearby(a: range, b: range)->bool:
    if a.start-b.stop > NEARBY_THRESHOLD:
        return False
    if b.start-a.stop > NEARBY_THRESHOLD:
        return False
    return True
