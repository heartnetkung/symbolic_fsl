from ..base import *
from ..graphic import *
from copy import deepcopy
from collections.abc import Iterable

MAX_COLORLESS_SHAPE_MASS = 100


class LoopStopper:
    def __init__(self)->None:
        self.count = 0

    def should_stop(self)->bool:
        self.count += 1
        return self.count >= 100000


def recursive_shape_split(
        shape: Shape, subshapes: Iterable[Shape],
        colorless: bool = False, transform: bool = False)->Optional[list[Shape]]:
    '''
    Split the given shape into subshapes such that
    all pixels belong to one and only one subshape.
    '''
    subshape_grids = [comp._grid.trim(True) for comp in subshapes]
    stopper = LoopStopper()

    if transform:
        subshape_grids_temp = []
        for comp_grid in subshape_grids:
            subshape_grids_temp += geom_transform_all(
                comp_grid, include_inverse=True)
        subshape_grids = subshape_grids_temp

    if colorless:
        original_shape = shape
        shape = Unknown(shape.x, shape.y, shape._grid.normalize_color())
        if _mass_larger_than(shape.grid, MAX_COLORLESS_SHAPE_MASS):
            return None
        subshape_grids = [grid.normalize_color() for grid in subshape_grids]
    else:
        original_shape = shape

    subshape_grids = _clean_subshapes(subshape_grids)
    return _recursive_shape_split(
        _trim(shape),  _trim(original_shape), subshape_grids, stopper)


def _recursive_shape_split(shape: Shape, original_shape: Shape, subshapes: list[Grid],
                           stopper: LoopStopper)->Optional[list[Shape]]:
    if stopper.should_stop():
        return None

    first_pixel = _find_first_pixel(shape)
    if first_pixel is None:
        return []  # success

    for subshape in subshapes:
        offset = shape._grid.find_subgrid(subshape)
        if offset is None:
            continue  # shape not found

        subtracted_shape = _subtract(shape, subshape, offset[0], offset[1])
        if first_pixel == _find_first_pixel(subtracted_shape):
            continue  # the first pixel needs to change

        subtracted_original = _subtract(original_shape, subshape, offset[0], offset[1])
        new_result = _recursive_shape_split(
            _trim(subtracted_shape), _trim(subtracted_original), subshapes, stopper)
        if new_result is None:
            continue  # depth-first fails

        # success
        new_x, new_y = shape.x+offset[0], shape.y+offset[1]
        return [_intersect(original_shape, subshape, offset[0], offset[1])]+new_result
    return None


def _clean_subshapes(subshapes: list[Grid])->list[Grid]:
    result = {}
    for subshape in subshapes:
        if _mass_larger_than(subshape, 1):
            result[repr(subshape)] = subshape
    return list(result.values())


def _mass_larger_than(grid: Grid, n: int)->bool:
    '''This is required because a subshape of mass 1 causes combinatorial explosion.'''
    mass = 0
    for i in range(grid.height):
        for j in range(grid.width):
            if not valid_color(grid.data[i][j]):
                continue
            if mass > n:
                return True
            mass += 1
    return False


def _subtract(shape: Shape, subshape: Grid, offset_x: int, offset_y: int)->Shape:
    subshape_unknown = Unknown(shape.x+offset_x, shape.y+offset_y, subshape)
    return subtract(shape, subshape_unknown)


def _trim(shape: Shape)->Shape:
    trimmed_grid = shape._grid.trim(True)
    width_dff = shape.width-trimmed_grid.width
    height_diff = shape.height-trimmed_grid.height
    return Unknown(shape.x+width_dff, shape.y+height_diff, trimmed_grid)


def _find_first_pixel(shape: Shape)->Optional[int]:
    if shape.height == 0:
        return None

    row = shape._grid.data[0]
    for i, color in enumerate(row):
        if color != NULL_COLOR:
            return i
    return None


def _intersect(larger: Shape, smaller: Grid, offset_x: int, offset_y: int)->Shape:
    result = make_grid(smaller.width, smaller.height)
    for i in range(smaller.height):
        for j in range(smaller.width):
            if smaller.data[i][j] != NULL_COLOR:
                result.data[i][j] = larger._grid.safe_access(j+offset_x, i+offset_y)
    return from_grid(larger.x+offset_x, larger.y+offset_y, result)
