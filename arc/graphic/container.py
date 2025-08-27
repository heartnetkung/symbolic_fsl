from .shape import *
from .types import *
from .util import *
from typing import Generator
from collections.abc import Iterable


class NonOverlapingContainer:
    '''
    A data structure to store shapes where no shape is overlapped.
    Note that for simplicity, all shapes are treated as rectangle defined by the bounding box.
    '''

    def __init__(self, width: int, height: int)->None:
        self.grid = make_grid(width, height)
        self.shapes: dict[int, Shape] = {}
        self.max_index = 0

    def __len__(self)->int:
        return len(self.shapes)

    def items(self)->Iterable[Shape]:
        return self.shapes.values()

    def add(self, shape: Shape)->bool:
        '''
        Add a shape and return True if successful.
        If there is an overlapping shape, the operation fails and return False.
        '''
        # zero pixel
        coord = next(_shape_pixel_loop(shape, self.grid), None)
        if coord is None:
            return False

        for x, y in _shape_pixel_loop(shape, self.grid):
            if self.grid.data[y][x] != NULL_COLOR:
                return False

        self.shapes[self.max_index] = shape
        for x, y in _shape_pixel_loop(shape, self.grid):
            self.grid.data[y][x] = self.max_index

        self.max_index += 1
        return True

    def query_overlap(self, shape: Shape)->list[Shape]:
        '''
        List all shapes in the collection that overlap with the given shape.
        The result is always sorted by the insertion index.
        '''
        return self.query(self.get_overlap(shape))

    def remove(self, shape: Shape)->bool:
        '''
        Remove a shape in the collection and return True if successful.
        If that object does not exist in the collection, return False.
        '''

        coord = next(_shape_pixel_loop(shape, self.grid), None)
        if coord is None:
            return False

        index = self.grid.data[coord[1]][coord[0]]
        if index == NULL_COLOR:
            return False

        found_shape = self.shapes[index]
        if found_shape != shape:
            return False

        del self.shapes[index]
        for x, y in _shape_pixel_loop(shape, self.grid):
            self.grid.data[y][x] = NULL_COLOR
        return True

    def get_overlap(self, shape: Shape)->list[int]:
        '''[low-level] Get all pixel_index associated with the shape.'''
        result = []
        for x, y in _shape_pixel_loop(shape, self.grid):
            value = self.grid.safe_access(x, y)
            if value >= 0:
                result.append(value)
        return result

    def query(self, pixel_index: list[int])->list[Shape]:
        '''[low-level] Lookup shapes from the given shape_indexes.'''
        return [self.shapes[i] for i in sorted(set(pixel_index))]


def _shape_pixel_loop(shape: Shape, grid: Grid)->Generator[tuple[int, int], None, None]:
    valid_x, valid_y = range(0, grid.width), range(0, grid.height)

    for x in range(shape.width):
        for y in range(shape.height):
            if shape._draw_cell(x, y) == NULL_COLOR:
                continue

            return_x, return_y = shape.x+x, shape.y+y
            if (return_x in valid_x) and (return_y in valid_y):
                yield (return_x, return_y)
