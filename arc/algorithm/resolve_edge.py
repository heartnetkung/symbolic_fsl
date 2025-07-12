from ..graphic import *
from ..base import *
from copy import deepcopy


def resolve_edge(smaller: Shape, larger: Shape, smaller_grid: Grid,
                 colorless: bool = False)->Optional[Shape]:
    '''Assume subshape or transformed_subshape relationship'''
    if smaller.width > larger.width:
        return None
    if smaller.height > larger.height:
        return None
    is_top = smaller.y == 0
    is_left = smaller.x == 0
    is_right = (smaller.x+smaller.width) == smaller_grid.width
    is_bottom = (smaller.y+smaller.height) == smaller_grid.height
    if not (is_top or is_left or is_right or is_bottom):
        return None

    if colorless:
        colors = smaller._grid.list_colors()-{NULL_COLOR}
        if len(colors) != 1:
            return None  # a new color is needed
        larger2 = Unknown(larger.x, larger.y, larger._grid.colorize(colors.pop()))
    else:
        larger2 = larger

    transformed_grids = geom_transform_all(larger2._grid)
    for transformed_grid in transformed_grids:
        result = _resolve_edge(smaller, transformed_grid)
        if result is not None:
            return result
    return None


def _resolve_edge(smaller: Shape, larger: Grid)->Optional[Shape]:
    offsets = larger.find_subgrid(smaller._grid)
    if offsets is None:
        return None
    return Unknown(smaller.x-offsets[0], smaller.y-offsets[1], larger)
