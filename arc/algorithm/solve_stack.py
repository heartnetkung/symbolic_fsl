from ..graphic import *
from ..base import *
from collections import deque
from .split_stack import *


def solve_stack(shapes: list[Shape], grid: Grid)->Optional[list[Shape]]:
    '''
    Merge multiple single-color shapes into multiple rectangles.
    '''

    shape_dict = _create_shape_dict(shapes, grid)
    if shape_dict is None:
        return None

    found, result = False, []
    for color, shapes in shape_dict.items():
        shapes2, found2 = _convert_shapes(shapes, grid, color)

        shapes2.sort(key=lambda a: a.y*100+a.x)
        shapes3, found3 = _merge_loop(shapes2, grid, color)
        shapes2.sort(key=lambda a: a.x*100+a.y)
        shapes4, found4 = _merge_loop(shapes2, grid, color)
        result += shapes3 if len(shapes3) <= len(shapes4) else shapes4
        found = True if (found2 or found3 or found4) else found
    return result if found else None


def _merge_loop(shapes: list[Shape], grid: Grid, color: int)->tuple[list[Shape], bool]:
    result, queue, found = [], deque(shapes), False
    while len(queue) > 0:
        merging_shape = queue.popleft()
        for i in range(len(queue)):
            current_shape = queue.popleft()
            new_shape = _merge_object(merging_shape, current_shape, grid, color)
            if new_shape is None:
                queue.append(current_shape)
            else:
                found = True
                merging_shape = new_shape
        result.append(merging_shape)
    return result, found


def _create_shape_dict(
        shapes: list[Shape], grid: Grid)->Optional[dict[int, list[Shape]]]:
    shape_dict = {}
    for shape in shapes:
        colors = shape._grid.list_colors()-{NULL_COLOR}
        if len(colors) > 1:
            return None
        if not grid.has_color(NULL_COLOR):
            return None

        color = colors.pop()
        shape_dict[color] = shape_dict.get(color, [])+[shape]
    return shape_dict


def _convert_shapes(shapes: list[Shape], grid: Grid,
                    color: int)->tuple[list[Shape], bool]:
    found, shapes2, shapes3 = False, [], []
    for i, shape in enumerate(shapes):
        rect = _to_rectangle(shape, grid, color)
        if rect is not None:
            found = True
            shapes2.append(rect)
        else:
            shapes2.append(shape)

    for shape in shapes2:
        splitted = split_stack(shape)
        if splitted is not None:
            found = True
            shapes3 += splitted
        else:
            shapes3.append(shape)
    return shapes3, found


def _to_rectangle(shape: Shape, grid: Grid,
                  color: int)->Optional[FilledRectangle]:
    if shape.__class__ == FilledRectangle:
        return None
    return _merge_object(shape, shape, grid, color)


def _merge_object(a: Shape, b: Shape, grid: Grid,
                  color: int)->Optional[FilledRectangle]:
    min_x, max_x = min(a.x, b.x), max(a.x+a.width, b.x+b.width)
    min_y, max_y = min(a.y, b.y), max(a.y+a.height, b.y+b.height)

    for i in range(min_y, max_y):
        for j in range(min_x, max_x):
            if grid.safe_access(j, i) in (NULL_COLOR, MISSING_VALUE):
                return None
    return FilledRectangle(
        min_x, min_y, max_x-min_x, max_y-min_y, color)
