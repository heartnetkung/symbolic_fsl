from ...base import *
from ...graphic import *
from copy import deepcopy
from ..util import *


class IntersectRectProcessor(ShapeConvProcess):
    def __init__(self, old_color: int, new_color: int, unchanged_color: int)->None:
        self.old_color = old_color
        self.new_color = new_color
        self.unchanged_color = unchanged_color
        self.validator_cache = {self.old_color, self.unchanged_color}

    def _can_start(self, grid: Grid, offset_x: int, offset_y: int)->bool:
        if grid.data[offset_y][offset_x] in self.validator_cache:
            return True
        return False

    def _to_result(self, grid: Grid, offset_x: int, offset_y: int,
                   w: int, h: int)->Optional[Shape]:
        result = make_grid(w, h)
        has_old_color, has_unchanged_color = False, False
        for x in range(w):
            for y in range(h):
                cell = grid.data[y+offset_y][x+offset_x]
                if cell == self.old_color:
                    has_old_color = True
                    result.data[y][x] = self.new_color
                elif cell == self.unchanged_color:
                    has_unchanged_color = True
                    result.data[y][x] = self.unchanged_color
                else:
                    raise Exception('should not happen')

        if has_old_color and has_unchanged_color:
            result2, offset_x2, offset_y2 = _trim_color(result, self.new_color)
            return Unknown(offset_x+offset_x2, offset_y+offset_y2, result2)
        return None

    def _can_expand(self, grid: Grid, offset_x: int, offset_y: int,
                    w: int, h: int, previous: bool, is_right: bool)->bool:
        if previous == False:
            return False
        if is_right and (offset_x+w+1 > grid.width):
            return False
        if (not is_right) and (offset_y+h+1 > grid.height):
            return False

        if is_right:
            to_check = {grid.safe_access(offset_x+w, i)
                        for i in range(offset_y, offset_y+h)}
        else:
            to_check = {grid.safe_access(j, offset_y+h)
                        for j in range(offset_x, offset_x+w)}
        return to_check.issubset(self.validator_cache)


def _trim_color(grid: Grid, color: int)->tuple[Grid, int, int]:
    x_min, y_min, x_max, y_max = 0, 0, grid.width, grid.height
    matching = {color}

    for y in range(grid.height):
        top_row = {grid.data[y][i] for i in range(grid.width)}
        if top_row != matching:
            break
        y_min += 1

    for y in range(grid.height-1, -1, -1):
        bottom_row = {grid.data[y][i] for i in range(grid.width)}
        if bottom_row != matching:
            break
        y_max -= 1

    for x in range(grid.width):
        left_row = {grid.data[i][x] for i in range(grid.height)}
        if left_row != matching:
            break
        x_min += 1

    for x in range(grid.width-1, -1, -1):
        right_row = {grid.data[i][x] for i in range(grid.height)}
        if right_row != matching:
            break
        x_max -= 1

    return grid.crop(x_min, y_min, x_max-x_min, y_max-y_min), x_min, y_min
