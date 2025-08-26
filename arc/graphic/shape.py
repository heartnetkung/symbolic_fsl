from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional
from .util import *
from .types import Grid
from functools import cached_property
from ..constant import NULL_COLOR, MISSING_VALUE
from enum import Enum


class ShapeType(Enum):
    filled_rectangle = 0
    hollow_rectangle = 1
    diagonal = 2
    unknown = 3
    shape = 4  # should not be used


class Shape(RuntimeObject):
    '''Anything drawable'''

    def __init__(self, x: int, y: int, width: int, height: int)->None:
        # x,y,width,height maybe -1 if ambiguous
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        super().__init__()

    @abstractmethod
    def _draw_cell(self, i: int, j: int)->int:
        '''
        Draw a cell to canvas with i starting from 0-height and j starting from 0-width
        If the color is not between 0 and 9, the draw is ignored.
        '''
        pass

    @cached_property
    def mass(self)->int:
        '''Count the number of valid draw.'''
        return np.sum(np.array(self._grid.data) != NULL_COLOR)

    @cached_property
    def top_color(self)->int:
        return self._grid.get_top_color()

    @cached_property
    def _grid(self)->Grid:
        result = make_grid(self.width, self.height)
        self.draw(result, False)
        return result

    @property
    def shape_type(self)->int:
        return ShapeType.shape.value

    @cached_property
    def shape_value(self)->int:
        return MISSING_VALUE

    @property
    def single_color(self)->int:
        top_color = self._grid.get_top_color()
        least_color = self._grid.get_least_color()
        if (top_color == NULL_COLOR) or (least_color == NULL_COLOR):
            return NULL_COLOR
        if top_color != least_color:
            return NULL_COLOR
        return top_color

    def draw(self, canvas: Grid, include_xy: bool = True)->None:
        '''Draw this object on canvas'''
        grid_height, grid_width = canvas.height, canvas.width
        for i in range(0, self.height):
            for j in range(0, self.width):
                if include_xy:
                    i2, j2 = i+self.y, j+self.x
                else:
                    i2, j2 = i, j

                if i2 >= 0 and j2 >= 0 and i2 < grid_height and j2 < grid_width:
                    color = self._draw_cell(i, j)
                    if valid_color(color):
                        canvas.data[i2][j2] = color

    def _base_entropy(self)->dict[str, Any]:
        '''Used when transforming into DataFrame.'''
        return {'class_': self.__class__.__name__, 'center_x': self.x+self.width/2,
                'center_y': self.y+self.height/2, 'height': self.height,
                'width': self.width}

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if not isinstance(other, Shape):
            return False
        return self.to_entropy_var() == other.to_entropy_var()

    def __lt__(self, obj)->bool:
        if self.__class__.__name__ != obj.__class__.__name__:
            return self.__class__.__name__ < obj.__class__.__name__
        self_vars, obj_vars = self.__dict__, obj.__dict__
        for key in sorted(self_vars.keys()):
            if self_vars[key] != obj_vars[key]:
                return self_vars[key] < obj_vars[key]
        return False


class FilledRectangle(Shape):
    def __init__(self, x: int, y: int, width: int, height: int, color: int)->None:
        super().__init__(x, y, width, height)
        self.color = color

    def _draw_cell(self, i: int, j: int)->int:
        return self.color

    def to_entropy_var(self)->dict[str, Any]:
        return super()._base_entropy() | {'color': self.color}

    def to_input_var(self)->dict[str, Any]:
        result = super().to_input_var()
        del result['color']
        return result

    @property
    def single_color(self)->int:
        return self.color

    @property
    def shape_type(self)->int:
        return ShapeType.filled_rectangle.value

    @staticmethod
    def is_valid(x: int, y: int, grid: Grid)->bool:
        grid_width, grid_height = grid.width, grid.height
        if grid_height == 0 or grid_width == 0:
            return False

        first_cell = grid.data[0][0]
        if first_cell == NULL_COLOR:
            return False

        for i in range(grid_height):
            for j in range(grid_width):
                if grid.data[i][j] != first_cell:
                    return False
        return True

    @staticmethod
    def from_grid(x: int, y: int, grid: Grid)->FilledRectangle:
        return FilledRectangle(x, y, grid.width, grid.height, grid.data[0][0])


class HollowRectangle(Shape):
    def __init__(self, x: int, y: int, width: int, height: int, color: int,
                 stroke: int)->None:
        super().__init__(x, y, width, height)
        assert (width > 2*stroke) and (height > 2*stroke)
        self.color = color
        self.stroke = stroke

    def _draw_cell(self, i: int, j: int)->int:
        if i < self.stroke or j < self.stroke:
            return self.color
        if i >= (self.height-self.stroke) or j >= (self.width-self.stroke):
            return self.color
        return NULL_COLOR

    def to_entropy_var(self)->dict[str, Any]:
        return super()._base_entropy() | {'color': self.color, 'stroke': self.stroke}

    def to_input_var(self)->dict[str, Any]:
        result = super().to_input_var()
        del result['color']
        return result

    @property
    def single_color(self)->int:
        return self.color

    @property
    def shape_type(self)->int:
        return ShapeType.hollow_rectangle.value

    @staticmethod
    def _get_stroke(grid: Grid)->tuple[int, int]:
        color, stroke = grid.data[0][0], 0
        for i in range(min(grid.width, grid.height)):
            if grid.data[i][i] != color:
                break
            stroke += 1
        return color, stroke

    @staticmethod
    def is_valid(x: int, y: int, grid: Grid)->bool:
        grid_width, grid_height = grid.width, grid.height
        if grid_height < 3 or grid_width < 3:
            return False

        first_cell, stroke = HollowRectangle._get_stroke(grid)
        if (grid_width <= 2*stroke) or (grid_height <= 2*stroke):
            return False

        for i in range(stroke, grid_height-stroke):
            for j in range(stroke, grid_width-stroke):
                if grid.data[i][j] != NULL_COLOR:
                    return False
        for i in range(grid_height):
            for k in range(stroke):
                if (grid.data[i][k] != first_cell or
                        grid.data[i][grid_width-1-k] != first_cell):
                    return False
        for j in range(grid_width):
            for k in range(stroke):
                if (grid.data[k][j] != first_cell or
                        grid.data[grid_height-1-k][j] != first_cell):
                    return False
        return True

    @staticmethod
    def from_grid(x: int, y: int, grid: Grid)->HollowRectangle:
        color, stroke = HollowRectangle._get_stroke(grid)
        return HollowRectangle(x, y, grid.width, grid.height, color, stroke)


class Diagonal(Shape):
    def __init__(self, x: int, y: int, width: int, color: int, north_west: bool)->None:
        super().__init__(x, y, width, width)
        self.color = color
        self.north_west = north_west

    def to_entropy_var(self)->dict[str, Any]:
        return super()._base_entropy() | {
            'color': self.color, 'north_west': self.north_west}

    def to_input_var(self)->dict[str, Any]:
        result = super().to_input_var()
        del result['color']
        # boolean casting
        result['north_west'] = 1 if result['north_west'] else 0
        return result

    def _draw_cell(self, i: int, j: int)->int:
        if self.north_west and (i == j):
            return self.color
        if (not self.north_west) and (i+j+1 == self.width):
            return self.color
        return NULL_COLOR

    @property
    def single_color(self)->int:
        return self.color

    @property
    def shape_type(self)->int:
        return ShapeType.diagonal.value

    @staticmethod
    def is_valid(x: int, y: int, grid: Grid)->bool:
        if grid.width != grid.height:
            return False

        height = grid.height
        if grid.data[0][0] != NULL_COLOR:
            color = grid.data[0][0]
            north_west = True
        elif grid.data[height-1][0] != NULL_COLOR:
            color = grid.data[grid.width-1][0]
            north_west = False
        else:
            return False

        if north_west:
            for i in range(height):
                for j in range(height):
                    cell = grid.data[i][j]
                    if i == j:
                        if cell != color:
                            return False
                    else:
                        if cell != NULL_COLOR:
                            return False
        else:
            for i in range(height):
                for j in range(height):
                    cell = grid.data[i][j]
                    if i+j+1 == height:
                        if cell != color:
                            return False
                    else:
                        if cell != NULL_COLOR:
                            return False
        return True

    @staticmethod
    def from_grid(x: int, y: int, grid: Grid)->Diagonal:
        if grid.data[0][0] != NULL_COLOR:
            color = grid.data[0][0]
            north_west = True
        elif grid.data[grid.width-1][0] != NULL_COLOR:
            color = grid.data[grid.width-1][0]
            north_west = False
        else:
            assert False, 'Unknown diagonal model'
        return Diagonal(x, y, grid.width, color, north_west)


class Unknown(Shape):
    def __init__(self, x: int, y: int, grid: Grid)->None:
        super().__init__(x, y, grid.width, grid.height)
        self.grid = grid

    def to_entropy_var(self)->dict[str, Any]:
        grid_width, grid_height = self.grid.width, self.grid.height
        to_append = {}
        for i in range(grid_height):
            for j in range(grid_width):
                to_append[f'[{i}][{j}]'] = self.grid.data[i][j]
        return super()._base_entropy() | to_append

    def _draw_cell(self, i: int, j: int)->int:
        return self.grid.data[i][j]

    @property
    def shape_type(self)->int:
        return ShapeType.unknown.value

    @property
    def shape_value(self)->int:  # type:ignore
        return hash(repr(self.grid.normalize_color()))

    @property
    def _grid(self)->Grid:  # type:ignore
        return self.grid

    def __repr__(self)->str:
        arr_lines = [repr(row) for row in self.grid.data]
        return '\n{}({}, {}, [\n{}])'.format(
            self.__class__.__name__, self.x, self.y, ',\n'.join(arr_lines))

    def to_input_var(self)->dict[str, Any]:
        result = super().to_input_var()
        del result['grid']
        return result

    @staticmethod
    def from_grid(x: int, y: int, grid: Grid)->Unknown:
        return Unknown(x, y, grid)

    @staticmethod
    def is_valid(x: int, y: int, grid: Grid)->bool:
        return True


NULL_SHAPE = Unknown(0, 0, Grid([[0]]))
