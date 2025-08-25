from .shape import *
from .types import *
from .util import *


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

    def add(self, shape: Shape)->bool:
        '''
        Add a shape and return True if successful.
        If there is an overlapping shape, the operation fails and return False.
        '''

        for x in range(shape.x, shape.x+shape.width):
            for y in range(shape.y, shape.y+shape.height):
                if self.grid.safe_access(x, y) != NULL_COLOR:
                    return False

        self.shapes[self.max_index] = shape
        for x in range(shape.x, shape.x+shape.width):
            for y in range(shape.y, shape.y+shape.height):
                self.grid.data[y][x] = self.max_index

        self.max_index += 1
        return True

    def query_overlap(self, shape: Shape)->list[Shape]:
        '''
        List all shapes in the collection that overlap with the given shape.
        The result is always sorted by the insertion index.
        '''

        result_indexes = set()
        for x in range(shape.x, shape.x+shape.width):
            for y in range(shape.y, shape.y+shape.height):
                value = self.grid.safe_access(x, y)
                if value >= 0:
                    result_indexes.add(value)

        return [self.shapes[i] for i in sorted(result_indexes)]

    def remove(self, shape: Shape)->bool:
        '''
        Remove a shape in the collection and return True if successful.
        If that object does not exist in the collection, return False.
        '''
        index = self.grid.safe_access(shape.x, shape.y)
        if index < 0:
            return False

        found_shape = self.shapes[index]
        if found_shape != shape:
            return False

        del self.shapes[index]
        for x in range(shape.x, shape.x+shape.width):
            for y in range(shape.y, shape.y+shape.height):
                self.grid.data[y][x] = NULL_COLOR
        return True
