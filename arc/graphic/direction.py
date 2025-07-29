from __future__ import annotations
from enum import Enum
from .shape import Shape
from .types import Coordinate
from typing import Optional


class Direction(Enum):
    # the numbers are intentional
    north = 0
    north_east = 1
    east = 2
    south_east = 3
    south = 4
    south_west = 5
    west = 6
    north_west = 7

    def get_offset(self)->tuple[int, int]:
        if self == Direction.north:
            return 0, -1
        elif self == Direction.north_east:
            return 1, -1
        elif self == Direction.east:
            return 1, 0
        elif self == Direction.south_east:
            return 1, 1
        elif self == Direction.south:
            return 0, 1
        elif self == Direction.south_west:
            return -1, 1
        elif self == Direction.west:
            return -1, 0
        elif self == Direction.north_west:
            return -1, -1
        else:
            raise Exception('unknown enum')

    def proceed(self, coord: Coordinate, distance: int = 1)->Coordinate:
        offset_x, offset_y = self.get_offset()
        return Coordinate(coord.x+(distance*offset_x), coord.y+(distance*offset_y))

    def left(self)->Direction:
        return Direction((self.value+6) % 8)

    def right(self)->Direction:
        return Direction((self.value+2) % 8)

    def is_diagonal(self)->bool:
        return self.value % 2 == 1

    @staticmethod
    def from_coord(before: Coordinate, after: Coordinate)->Optional[Direction]:
        offset = (after.x-before.x, after.y-before.y)
        for dir_ in Direction:
            if dir_.get_offset() == offset:
                return dir_
        return None

    @staticmethod
    def listing(diag: bool)->list[Direction]:
        if diag:
            return [Direction.north_west, Direction.north_east,
                    Direction.south_east, Direction.south_west]
        return [Direction.north, Direction.south, Direction.west, Direction.east]


def sort_shapes(shapes: list[Shape], dir_: Direction)->list[Shape]:
    if dir_ == Direction.north:
        def func(shape): return shape.y
    elif dir_ == Direction.east:
        def func(shape): return -shape.x
    elif dir_ == Direction.west:
        def func(shape): return shape.x
    elif dir_ == Direction.south:
        def func(shape): return -shape.y
    else:
        raise Exception('unsupported type')
    return sorted(shapes, key=func)
