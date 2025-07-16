from __future__ import annotations
from ...base import *
from ...graphic import *
from enum import Enum
from dataclasses import dataclass


class Navigation(Enum):
    proceed = 0
    turn_left = 1
    turn_right = 2
    stop = 3


class Direction(Enum):
    # intentionally clockwise
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


@dataclass(frozen=True)
class Line:
    x: int
    y: int
    width: int
    height: int
    color: int
    coords: list[Coordinate]

    @staticmethod
    def make(coords: list[Coordinate], color: int)->Optional[Line]:
        if len(coords) == 0:
            return None

        x_coords = [coord.x for coord in coords]
        y_coords = [coord.y for coord in coords]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width, height = max_x-min_x+1, max_y-min_y+1
        return Line(min_x, min_y, width, height, color, coords)

    def __post_init__(self):
        assert len(self.coords) > 0

    def get_start_end_pixels(self)->tuple[FilledRectangle, FilledRectangle]:
        start_coord, final_coord = self.coords[0], self.coords[-1]
        start_pixel = FilledRectangle(start_coord.x, start_coord.y, 1, 1, self.color)
        end_pixel = FilledRectangle(final_coord.x, final_coord.y, 1, 1, self.color)
        return start_pixel, end_pixel

    def to_shape(self)->Shape:
        if self.width == 1 or self.height == 1:
            return FilledRectangle(
                self.x, self.y, self.width, self.height, self.color)
        if self.width == self.height == len(self.coords):
            north_west = self.coords[0] == Coordinate(self.x, self.y)
            return Diagonal(self.x, self.y, self.width, self.color, north_west)

        canvas = make_grid(self.width, self.height)
        for coord in self.coords:
            canvas.safe_assign(coord.x-self.x, coord.y-self.y, self.color)
        return Unknown(self.x, self.y, canvas)

    def reverse(self)->Line:
        coords2 = self.coords.copy()
        coords2.reverse()
        return Line(self.x, self.y, self.width, self.height, self.color, coords2)

    def to_dir(self)->Optional[tuple[Direction, list[Navigation]]]:
        if len(self.coords) == 1:
            return None

        init_dir = Direction.from_coord(self.coords[0], self.coords[1])
        if init_dir is None:
            return None

        current_dir = init_dir
        current_coord = self.coords[0]
        navs = []

        for new_coord in self.coords[1:]:
            new_dir = Direction.from_coord(current_coord, new_coord)
            if new_dir is None:
                return None

            if new_dir == current_dir:
                navs.append(Navigation.proceed)
            elif new_dir == current_dir.left():
                navs.append(Navigation.turn_left)
            elif new_dir == current_dir.right():
                navs.append(Navigation.turn_right)
            else:
                return None

            current_dir = new_dir
            current_coord = new_coord

        navs.append(Navigation.stop)
        return init_dir, navs

    def has_turn(self)->bool:
        dir_blob = self.to_dir()
        if dir_blob is None:
            return False

        _, navs = dir_blob
        for nav in navs:
            if nav in (Navigation.turn_left, Navigation.turn_right):
                return True
        return False
