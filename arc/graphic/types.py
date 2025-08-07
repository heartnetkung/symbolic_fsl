from __future__ import annotations
from copy import deepcopy
from ..constant import NULL_COLOR, MISSING_VALUE
from dataclasses import dataclass
import math
from typing import Optional
from functools import cached_property
from collections import Counter

EPSILON = 0.01


class Grid:
    '''mutable 2d list of int with utility methods'''

    def __init__(self, data: list[list[int]])->None:
        self.data = data

    def __eq__(self, other)->bool:
        if type(other) != Grid:
            return False
        return self.data == other.data

    def __hash__(self)->int:
        return hash(repr(self.data))

    def __repr__(self)->str:
        return repr(self.data)

    def offset_subshape(self, smaller: Grid, offset_x: int, offset_y: int)->bool:
        assert smaller.width+offset_x <= self.width
        assert smaller.height+offset_y <= self.height
        for i in range(smaller.height):
            for j in range(smaller.width):
                smaller_cell = smaller.data[i][j]
                self_cell = self.data[i+offset_y][j+offset_x]
                if self_cell != smaller_cell and smaller_cell != NULL_COLOR:
                    return False
        return True

    def offset_subshape_approx(self, smaller: Grid, offset_x: int, offset_y: int,
                               threshold: float)->float:
        assert smaller.width+offset_x <= self.width
        assert smaller.height+offset_y <= self.height
        total, fail_count = smaller.height*smaller.width, 0

        for i in range(smaller.height):
            for j in range(smaller.width):
                smaller_cell = smaller.data[i][j]
                self_cell = self.data[i+offset_y][j+offset_x]
                if self_cell != smaller_cell and smaller_cell != NULL_COLOR:
                    fail_count += 1
                    if fail_count/total >= (1-threshold):
                        return 0
        return (total-fail_count)/total

    @property
    def width(self)->int:
        if len(self.data) == 0:
            return 0
        return len(self.data[0])

    @property
    def height(self)->int:
        return len(self.data)

    def safe_access(self, x: int, y: int)->int:
        '''access without error'''
        if x < 0 or y < 0:
            return MISSING_VALUE
        if x >= self.width or y >= self.height:
            return MISSING_VALUE
        return self.data[y][x]

    def safe_access_c(self, coord: Coordinate)->int:
        return self.safe_access(coord.x, coord.y)

    def safe_assign(self, x: int, y: int, value: int)->None:
        if self.safe_access(x, y) != MISSING_VALUE:
            self.data[y][x] = value

    def safe_assign_c(self, coord: Coordinate, value: int)->None:
        if self.safe_access(coord.x, coord.y) != MISSING_VALUE:
            self.data[coord.y][coord.x] = value

    def draw(self, dest_grid: Grid, offset_x: int, offset_y: int)->None:
        if (self.height == 0) or (dest_grid.height == 0):
            return

        x = range_intersect(
            range(0, self.width), range(-offset_x, dest_grid.width-offset_x))
        y = range_intersect(
            range(0, self.height), range(-offset_y, dest_grid.height-offset_y))

        for i in y:
            for j in x:
                dest_grid.data[i+offset_y][j+offset_x] = self.data[i][j]

    @cached_property
    def color_count(self)->Counter[int]:
        result = Counter()
        for row in self.data:
            to_add = [cell for cell in row if cell != NULL_COLOR]
            result.update(to_add)
        return result

    def get_top_color(self)->int:
        color_ranks = self.color_count.most_common(1)
        if len(color_ranks) == 0:
            return NULL_COLOR
        return color_ranks[0][0]

    def get_second_top_color(self)->int:
        color_ranks = self.color_count.most_common(2)
        if len(color_ranks) < 2:
            return NULL_COLOR
        return color_ranks[1][0]

    def get_least_color(self)->int:
        color_ranks = self.color_count.most_common()
        if len(color_ranks) == 0:
            return NULL_COLOR
        return color_ranks[-1][0]

    def print_grid(self)->None:
        for row in self.data:
            print(row)

    def print_grid2(self)->None:
        print('Grid([')
        _last = len(self.data)-1
        for i, row in enumerate(self.data):
            if i != _last:
                print(row, ',')
            else:
                print(row)
        print('])')

    def replace_color(self, old_color: int, new_color: int)->Grid:
        result = deepcopy(self.data)
        for i in range(self.height):
            for j in range(self.width):
                if result[i][j] == old_color:
                    result[i][j] = new_color
        return Grid(result)

    def keep_color(self, color: int)->Grid:
        result = deepcopy(self.data)
        for i in range(self.height):
            for j in range(self.width):
                if result[i][j] != color:
                    result[i][j] = NULL_COLOR
        return Grid(result)

    def list_colors(self)->set[int]:
        result = set()
        for row in self.data:
            for cell in row:
                result.add(cell)
        return result

    def trim(self, top_left_only: bool = False)->Grid:
        min_y, found = 0, False
        for i in range(self.height):
            for j in range(self.width):
                if self.data[i][j] != NULL_COLOR:
                    found = True
                    break
            if found:
                break
            min_y += 1

        min_x, found = 0, False
        for j in range(self.width):
            for i in range(self.height):
                if self.data[i][j] != NULL_COLOR:
                    found = True
                    break
            if found:
                break
            min_x += 1

        if top_left_only:
            return self.crop(min_x, min_y, self.width-min_x, self.height-min_y)

        max_y, found = self.height, False
        for i in range(self.height-1, -1, -1):
            for j in range(self.width):
                if self.data[i][j] != NULL_COLOR:
                    found = True
                    break
            if found:
                break
            max_y -= 1

        max_x, found = self.width, False
        for j in range(self.width-1, -1, -1):
            for i in range(self.height):
                if self.data[i][j] != NULL_COLOR:
                    found = True
                    break
            if found:
                break
            max_x -= 1
        return self.crop(min_x, min_y, max_x-min_x, max_y-min_y)

    def crop(self, x: int, y: int, w: int, h: int)->Grid:
        result = []
        for i in range(y, y+h):
            result.append(self.data[i][x:x+w].copy())
        return Grid(result)

    def transpose(self)->Grid:
        return Grid([[self.data[i][j] for i in range(self.height)]
                     for j in range(self.width)])

    def flip_h(self)->Grid:
        return Grid([[self.data[-i-1][j]for j in range(self.width)]
                     for i in range(self.height)])

    def flip_v(self)->Grid:
        return Grid([[self.data[i][-j-1]for j in range(self.width)]
                     for i in range(self.height)])

    def flip_both(self)->Grid:
        return Grid([[self.data[-i-1][-j-1]for j in range(self.width)]
                     for i in range(self.height)])

    def remove_bg(self)->Grid:
        return self.replace_color(self.get_top_color(), NULL_COLOR)

    def normalize_color(self)->Grid:
        return Grid([[1 if cell != NULL_COLOR else NULL_COLOR for cell in row]
                     for row in self.data])

    def colorize(self, new_color: int)->Grid:
        return Grid([[new_color if cell != NULL_COLOR else NULL_COLOR for cell in row]
                     for row in self.data])

    def scale_up(self, x_scale: int, y_scale: int)->Grid:
        result = [[-1 for j in range(self.width*x_scale)]
                  for i in range(self.height*y_scale)]
        for i in range(self.height*y_scale):
            for j in range(self.width*x_scale):
                result[i][j] = self.data[
                    math.floor(i/y_scale)][math.floor(j/x_scale)]
        return Grid(result)

    def has_color(self, color: int)->bool:
        for i in range(self.height):
            for j in range(self.width):
                if self.data[i][j] == color:
                    return True
        return False

    def find_subgrid(self, smaller_grid: Grid)->Optional[tuple[int, int]]:
        width_diff = self.width - smaller_grid.width
        height_diff = self.height - smaller_grid.height

        for offset_y in range(height_diff+1):
            for offset_x in range(width_diff+1):
                if self.offset_subshape(smaller_grid, offset_x, offset_y):
                    return offset_x, offset_y
        return None

    def find_subgrid_approx(self, smaller_grid: Grid,
                            threshold: float)->Optional[tuple[int, int]]:
        width_diff = self.width - smaller_grid.width
        height_diff = self.height - smaller_grid.height

        for offset_y in range(height_diff+1):
            for offset_x in range(width_diff+1):
                if self.offset_subshape_approx(
                        smaller_grid, offset_x, offset_y, threshold) < EPSILON:
                    return offset_x, offset_y
        return None

    def inverse(self)->Optional[Grid]:
        colors = self.list_colors()
        if len(colors) != 2 or NULL_COLOR not in colors:
            return None

        color = (colors-{NULL_COLOR}).pop()
        return Grid([[color if cell == NULL_COLOR else NULL_COLOR for cell in row]
                     for row in self.data])


@dataclass(frozen=True)
class Coordinate:
    x: int
    y: int

    def x_to_y(self, offset_x: int, offset_y: int)->Coordinate:
        return Coordinate(self.x+offset_x, self.y+offset_y)

    def y_to_x(self, offset_x: int, offset_y: int)->Coordinate:
        return Coordinate(self.x-offset_x, self.y-offset_y)


@dataclass(frozen=True)
class FloatCoordinate:
    x: float
    y: float


def range_intersect(r1: range, r2: range)->range:
    if (r1.start > r2.stop) or (r2.start > r1.stop):
        return range(0, 0)
    return range(max(r1.start, r2.start), min(r1.stop, r2.stop))



def find_separators(grid: Grid, color: int = NULL_COLOR)->tuple[
        list[int], list[int], list[int], list[int]]:
    rows, row_colors = _find_row_separator(grid, color)
    cols, col_colors = _find_row_separator(grid.transpose(), color)
    return rows, cols, row_colors, col_colors


def _find_row_separator(grid: Grid, color: int = NULL_COLOR
                        )->tuple[list[int], list[int]]:
    rows, row_colors = [], []
    for row in range(grid.height):
        unique_el = {grid.data[row][i] for i in range(grid.width)}
        first_el = unique_el.pop()
        if len(unique_el) != 0 or first_el == NULL_COLOR:
            continue
        if color == NULL_COLOR or first_el == color:
            rows.append(row)
            row_colors.append(first_el)
    return rows+[grid.height], row_colors
