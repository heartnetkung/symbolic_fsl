from __future__ import annotations
from ..graphic import *
from ..base import *
from copy import deepcopy
from dataclasses import dataclass

HIDDEN_COLOR = -2


def split_stack(a: Shape)->Optional[list[Shape]]:
    '''
    Split a single-color shape into multiple rectangles.
    Used by merge_stack.
    '''

    if a.__class__ == FilledRectangle:
        return None

    offset_x, offset_y = a.x, a.y
    results, current_grid = [], deepcopy(a._grid)

    for i in range(a.height):
        if current_grid.width == 0 or current_grid.height == 0:
            return None

        # initialize all partial rectangles found in the first row
        first_row = current_grid.data[0]
        rectangles = _create_PartialRectangle(first_row)

        # greedily expand all rectangles
        for j in range(1, current_grid.height):
            row = current_grid.data[j]
            for k, rectangle in enumerate(rectangles):
                success = rectangle.expand(row)

        # collect result
        for rectangle in rectangles:
            results.append(rectangle.to_shape(offset_x, offset_y))
            _remove_color(current_grid, rectangle)

        # trim and update data for the next iteration
        _remove_first_line(current_grid)
        next_grid = current_grid.trim(True)
        offset_x += current_grid.width-next_grid.width
        offset_y += current_grid.height-next_grid.height
        current_grid = next_grid

    return results if len(results) > 0 else None


def _create_PartialRectangle(row: list[int])->list[_PartialRectangle]:
    result, offset = [], -1
    for min_x in range(len(row)):
        if min_x <= offset:
            continue

        color = row[min_x]
        if valid_color(color):
            potential_min_x = min_x
            for i in range(min_x-1, -1, -1):
                if row[i] not in (color, HIDDEN_COLOR):
                    break
                potential_min_x -= 1

            potential_max_x = min_x
            for i in range(min_x+1, len(row)):
                if row[i] not in (color, HIDDEN_COLOR):
                    break
                potential_max_x += 1

            max_x = potential_max_x
            for i in range(potential_max_x, min_x, -1):
                if row[i] == color:
                    break
                max_x -= 1

            y, height = 0, 1
            offset = potential_max_x
            result.append(_PartialRectangle(
                min_x, max_x, potential_min_x, potential_max_x, y, height, color))
    return result


def _remove_color(grid: Grid, rectangle: _PartialRectangle)->None:
    for i in range(rectangle.y+1, rectangle.y+rectangle.height):
        for j in range(rectangle.min_x, rectangle.max_x+1):
            grid.data[i][j] = HIDDEN_COLOR


def _remove_first_line(grid: Grid)->None:
    for i in range(grid.width):
        grid.data[0][i] = NULL_COLOR


@dataclass
class _PartialRectangle:
    min_x: int
    max_x: int
    potential_min_x: int
    potential_max_x: int
    y: int
    height: int
    color: int
    done: bool = False

    def to_shape(self, offset_x: int, offset_y: int)->FilledRectangle:
        return FilledRectangle(
            offset_x+self.min_x, self.y+offset_y, self.max_x-self.min_x+1,
            self.height, self.color)

    def expand(self, row: list[int])->None:
        if self.done:
            return

        for i in range(self.min_x, self.max_x+1):
            if row[i] not in (self.color, HIDDEN_COLOR):
                self.done = True
                return
        for i in range(self.max_x+1, self.potential_max_x+1):
            if row[i] == self.color:
                self.max_x += 1
        for i in range(self.min_x-1, self.potential_min_x-1, -1):
            if row[i] == self.color:
                self.min_x -= 1
        self.height += 1
