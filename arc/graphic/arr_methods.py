from .shape import Shape, FilledRectangle
from copy import deepcopy
from .util import *
import numpy as np
from ..constant import NULL_COLOR
from .types import range_intersect


def bound_width(shapes: list[Shape])->int:
    if len(shapes) == 0:
        return -1

    max_x = max([shape.width+shape.x for shape in shapes])
    min_x = min([shape.x for shape in shapes])
    return max_x - min_x


def bound_height(shapes: list[Shape])->int:
    if len(shapes) == 0:
        return -1

    max_y = max([shape.height+shape.y for shape in shapes])
    min_y = min([shape.y for shape in shapes])
    return max_y - min_y


def bound_x(shapes: list[Shape])->int:
    if len(shapes) == 0:
        return -1
    return min([shape.x for shape in shapes])


def bound_y(shapes: list[Shape])->int:
    if len(shapes) == 0:
        return -1
    return min([shape.y for shape in shapes])


def total_mass(shapes: list[Shape])->int:
    canvas_width = max([shape.x+shape.width for shape in shapes])
    canvas_height = max([shape.y+shape.height for shape in shapes])
    canvas = make_grid(canvas_width, canvas_height)
    for shape in shapes:
        shape.draw(canvas)
    return (np.array(canvas.data) != NULL_COLOR).sum()


def top_mass(shapes: list[Shape])->int:
    if len(shapes) == 0:
        return 0
    return max(shapes, key=lambda x: x.mass).mass


def duplicates(prototypes: list[Shape], new_coords: list[tuple[int, int]])->list[Shape]:
    result = []
    for prototype in prototypes:
        for x, y in new_coords:
            new_shape = deepcopy(prototype)
            new_shape.x += x
            new_shape.y += y
            result.append(new_shape)
    return result


def find_inner_bound(a: Shape, b: Shape)->FilledRectangle:
    inner_x = _find_inner_bound_1d(range(a.x, a.x+a.width), range(b.x, b.x+b.width))
    inner_y = _find_inner_bound_1d(range(a.y, a.y+a.height), range(b.y, b.y+b.height))
    return FilledRectangle(inner_x.start, inner_y.start,
                           inner_x.stop-inner_x.start, inner_y.stop-inner_y.start, 0)


def _find_inner_bound_1d(a: range, b: range)->range:
    intersect_ = range_intersect(a, b)
    if len(intersect_) > 0:
        return intersect_
    if a.start < b.start:
        return range(a.stop, b.start)
    else:
        return range(b.stop, a.start)
