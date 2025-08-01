from ..base import *
from ..graphic import *
from copy import deepcopy

MAX_COLORLESS_SHAPE_MASS = 100


class LoopStopper:
    def __init__(self)->None:
        self.count = 0

    def should_stop(self)->bool:
        self.count += 1
        return self.count >= 100000


def recursive_shape_split(
        shape: Shape, subshapes: list[Shape],
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
        shape = Unknown(shape.x, shape.y, shape._grid.normalize_color())
        if _mass_larger_than(shape.grid, MAX_COLORLESS_SHAPE_MASS):
            return None

        subshape_grids_original = subshape_grids
        subshape_grids = [grid.normalize_color() for grid in subshape_grids]
    else:
        subshape_grids_original = subshape_grids

    subshape_grids, subshape_grids_original = _clean_subshapes(
        subshape_grids, subshape_grids_original)
    return _recursive_shape_split(
        _trim(shape), subshape_grids, stopper, subshape_grids_original)


def _recursive_shape_split(shape: Shape, subshapes: list[Grid], stopper: LoopStopper,
                           subshapes_original: list[Grid])->Optional[list[Shape]]:
    if stopper.should_stop():
        return None

    first_pixel = _find_first_pixel(shape)
    if first_pixel is None:
        return []  # success

    for subshape, subshape_original in zip(subshapes, subshapes_original):
        offset = shape._grid.find_subgrid(subshape)
        if offset is None:
            continue  # shape not found

        subtracted_shape = _subtract(shape, subshape, offset[0], offset[1])
        if first_pixel == _find_first_pixel(subtracted_shape):
            continue  # the first pixel needs to change

        new_container = _trim(subtracted_shape)
        new_result = _recursive_shape_split(
            new_container, subshapes, stopper, subshapes_original)
        if new_result is None:
            continue  # depth-first fails

        # success
        new_x, new_y = shape.x+offset[0], shape.y+offset[1]
        return [from_grid(new_x, new_y, subshape_original)]+new_result
    return None


def _clean_subshapes(subshapes: list[Grid],
                     original_subshapes: list[Grid])->tuple[list[Grid], list[Grid]]:
    assert len(subshapes) == len(original_subshapes)

    result = {}
    for subshape, orig_subshape in zip(subshapes, original_subshapes):
        if not _mass_larger_than(subshape, 1):
            continue
        result[repr(subshape)] = (subshape, orig_subshape)
    return [pair[0] for pair in result.values()], [pair[1] for pair in result.values()]


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
