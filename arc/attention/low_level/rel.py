from ...graphic import *
from .prop import *
import math
import numpy as np
from ...constant import *


def list_relationship(a: Shape, b: Shape, include_properties=True)->set[str]:
    result = set()
    a_dict, b_dict = list_properties(a), list_properties(b)
    if include_properties:
        for k, v in a_dict.items():
            if b_dict.get(k) == v:
                result.add(f'same_{k}')

    result |= touch_overlap(a, b)
    result |= _color_intersect(a_dict, b_dict)
    result |= _exact_contain(a, b)
    result |= _subshape(a, b)
    result |= _unknown_scale(a, b, a_dict, b_dict)
    result |= _x_or_y(a, b)
    result |= _full_overlap(a, b)
    result |= is_contain(a, b)
    return result


def _x_or_y(a: Shape, b: Shape)->set[str]:
    result = set()
    if a.x == b.x or a.y == b.y:
        result.add('same_x_or_y')

    range_ax, range_ay = range(a.x, a.x+a.width), range(a.y, a.y+a.height)
    range_bx, range_by = range(b.x, b.x+b.width), range(b.y, b.y+b.height)
    if _is_subrange(range_ax, range_bx):
        result.add('subset_x')
        result.add('subset_x_or_y')
    if _is_subrange(range_ay, range_by):
        result.add('subset_y')
        result.add('subset_x_or_y')
    return result


def is_contain(a: Shape, b: Shape)->set[str]:
    a_x2, a_y2 = a.x+a.width, a.y+a.height
    b_x2, b_y2 = b.x+b.width, b.y+b.height

    if ((a.x == b.x) and (a.y == b.y) and (a_x2 == b_x2) and (a_y2 == b_y2)):
        return set()
    if ((a.x <= b.x) and (a.y <= b.y) and (a_x2 >= b_x2) and (a_y2 >= b_y2)):
        return {'contain'}
    if ((b.x <= a.x) and (b.y <= a.y) and (b_x2 >= a_x2) and (b_y2 >= a_y2)):
        return {'contain'}
    return set()


def _is_subrange(a: range, b: range)->bool:
    assert len(a) > 0 and len(b) > 0
    return (((a.start in b) and (a[-1] in b)) or
            (b.start in a) and b[-1] in a)


def _color_intersect(a_dict: dict[str, int], b_dict: dict[str, int])->set[str]:
    intersect: set[int] = a_dict['colors'] & b_dict['colors']  # type:ignore
    if len(intersect) == 0:
        return set()
    return {'color_intersect'}


def touch_overlap(a: Shape, b: Shape)->set[str]:
    union_shape = union(a, b)
    mass_difference = a.mass + b.mass - union_shape.mass
    if mass_difference > 0:
        return {'overlap'}
    if len(list_sparse_objects(union_shape._grid)) != 1:
        return set()
    return {'touch'}


def _exact_contain(a: Shape, b: Shape)->set[str]:
    if a.width-1 > b.width and a.height-1 > b.height:
        larger_shape, smaller_shape = a, b
    elif b.width-1 > a.width and b.height-1 > a.height:
        larger_shape, smaller_shape = b, a
    else:
        return set()

    # make enclosed shapes
    canvas = make_grid(larger_shape.width, larger_shape.height)
    larger_shape.draw(canvas, include_xy=False)
    _flip_color(canvas)
    areas = list_objects(canvas, False)
    enclosed_shapes = _remove_edge_objects([areas], [canvas])[0]

    # comparison
    smaller_shape_values = list_shape_representations(smaller_shape)
    for enclosed_shape in enclosed_shapes:
        enclosed_shape_values = list_shape_representations(enclosed_shape)
        if (smaller_shape_values['colorless_shape'] ==
                enclosed_shape_values['colorless_shape']):
            return {'exact_transformed_contain', 'exact_contain'}
        if (smaller_shape_values['colorless_transformed_shape'] ==
                enclosed_shape_values['colorless_transformed_shape']):
            return {'exact_transformed_contain'}
    return set()


def _flip_color(grid: Grid)->None:
    for i in range(grid.height):
        for j in range(grid.width):
            if grid.data[i][j] == NULL_COLOR:
                grid.data[i][j] = 1
            else:
                grid.data[i][j] = NULL_COLOR


def _subshape(a: Shape, b: Shape)->set[str]:
    mass_a, mass_b = a.mass, b.mass
    if mass_a == mass_b:
        return set()

    larger = a if mass_a > mass_b else b
    smaller = b if mass_a > mass_b else a
    if (larger.width < smaller.width or larger.height < smaller.height):
        return set()

    if larger._grid.find_subgrid(smaller._grid) is not None:
        return {'subshape', 'transformed_subshape'}
    for transformed_grid in geom_transform_all(smaller._grid, True):
        if larger._grid.find_subgrid(transformed_grid) is not None:
            return {'transformed_subshape'}
    return set()


def _unknown_scale(a: Shape, b: Shape, a_dict: dict[str, int],
                   b_dict: dict[str, int])->set[str]:
    if a.__class__ != Unknown or b.__class__ != Unknown:
        return set()
    if a.width == 0 or a.height == 0 or b.width == 0 or b.height == 0:
        return set()

    width_ratio = a.width / b.width
    height_ratio = a.height / b.height
    if not np.isclose(width_ratio, height_ratio):
        return set()

    lcm = math.lcm(a.width, b.width)
    scale_a, scale_b = round(lcm/a.width), round(lcm/b.width)
    if a._grid.scale_up(scale_a, scale_a) != b._grid.scale_up(scale_b, scale_b):
        return set()
    return {'unknown_scale'}


def _remove_edge_objects(all_y_shapes: list[list[Shape]],
                         y_grids: list[Grid])->list[list[Shape]]:
    result = []
    for y_shapes, y_grid in zip(all_y_shapes, y_grids):
        filtered = [y_shape for y_shape in y_shapes if (
            (y_shape.x > 0) and (y_shape.y > 0) and
            (y_shape.x+y_shape.width < y_grid.width) and
            (y_shape.y+y_shape.height < y_grid.height))]
        result.append(filtered)
    return result


def _full_overlap(a: Shape, b: Shape)->set[str]:
    mass_a, mass_b = a.mass, b.mass
    if mass_a == mass_b:
        return set()

    larger = a if mass_a > mass_b else b
    smaller = b if mass_a > mass_b else a
    if (larger.width < smaller.width or larger.height < smaller.height):
        return set()

    subtracted = subtract(larger, smaller)
    if (larger.mass - subtracted.mass) != smaller.mass:
        return set()

    return {'full_overlap'}
