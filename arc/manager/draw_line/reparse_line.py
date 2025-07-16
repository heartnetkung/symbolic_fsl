from ...base import *
from ...graphic import *
from dataclasses import dataclass
from .types import *
from copy import deepcopy

MAX_LINE_PER_UNKNOWN = 30
MAX_LINE_LENGTH = 1000
USED_COLOR = -2


def reparse_line(shape: Shape)->list[Line]:
    if isinstance(shape, Diagonal):
        return _parse_diag(shape)
    if isinstance(shape, FilledRectangle):
        return _parse_rect(shape)
    if isinstance(shape, Unknown):
        return _parse_unknown(shape)
    return []


def _parse_diag(diag: Diagonal)->list[Line]:
    if diag.north_west:
        coords = [Coordinate(i+diag.x, i+diag.y) for i in range(diag.width)]
    else:
        y_end = diag.y+diag.height-1
        coords = [Coordinate(i+diag.x, y_end-i) for i in range(diag.width)]
    if len(coords) == 0:
        return []
    return [Line(diag.x, diag.y, diag.width, diag.width, diag.color, coords)]


def _parse_rect(rect: FilledRectangle)->list[Line]:
    if rect.width == 1:
        coords = [Coordinate(rect.x, rect.y+i) for i in range(rect.height)]
    elif rect.height == 1:
        coords = [Coordinate(rect.x+i, rect.y) for i in range(rect.width)]
    else:
        return []
    if len(coords) == 0:
        return []
    return [Line(rect.x, rect.y, rect.width, rect.height, rect.color, coords)]


def _parse_unknown(shape: Unknown)->list[Line]:
    colors = shape.grid.list_colors()-{NULL_COLOR}
    if len(colors) != 1:
        return []

    result, color = [], colors.pop()
    temp_grid = Grid(deepcopy(shape.grid.data))
    diag = _is_diag(temp_grid)
    for _ in range(MAX_LINE_PER_UNKNOWN):
        new_line = _parse_single_unknown(temp_grid, color, shape.x, shape.y, diag)
        if new_line is None:
            break
        result.append(new_line)

    left_colors = temp_grid.list_colors()
    if len(left_colors-{NULL_COLOR, USED_COLOR}) == 0:
        return result
    return []


def _parse_single_unknown(
        grid: Grid, color: int, x: int, y: int, diag: bool)->Optional[Line]:
    starting_blob = _find_start(grid, diag)
    if starting_blob is None:
        return None

    current_dir, init_x, init_y = starting_blob
    current_coord = Coordinate(init_x, init_y)
    coords = []

    for _ in range(MAX_LINE_LENGTH):
        coords.append(current_coord)
        grid.safe_assign_c(current_coord, USED_COLOR)
        left_dir, right_dir = current_dir.left(), current_dir.right()

        if _check_ahead(current_coord, grid, current_dir):
            pass  # keep current_dir
        elif _check_ahead(current_coord, grid, left_dir):
            current_dir = left_dir
        elif _check_ahead(current_coord, grid, right_dir):
            current_dir = right_dir
        else:
            break
        current_coord = current_dir.proceed(current_coord)

    global_coords = [Coordinate(x+coord.x, y+coord.y) for coord in coords]
    return Line.make(global_coords, color)


def _find_start(grid: Grid, diag: bool)->Optional[tuple[Direction, int, int]]:
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.safe_access(x, y)
            if not valid_color(cell):
                continue

            count = 0
            found_dir = Direction.north  # DUMMY
            for dir_ in Direction.listing(diag):
                offset_x, offset_y = dir_.get_offset()
                neighbor = grid.safe_access(x+offset_x, y+offset_y)
                if neighbor not in (NULL_COLOR, MISSING_VALUE):
                    count += 1
                    found_dir = dir_
            if count == 1:
                return (found_dir, x, y)
    return None


def _check_ahead(coord: Coordinate, grid: Grid, dir_: Direction)->bool:
    current_coord = coord
    for _ in range(MAX_LINE_LENGTH):
        current_coord = dir_.proceed(current_coord)
        current_cell = grid.safe_access_c(current_coord)
        if current_cell in (NULL_COLOR, MISSING_VALUE):
            return False
        if valid_color(current_cell):
            return True
        # keep continuing for USED_VALUE
    raise Exception('should not reach here')


def _is_diag(grid: Grid)->bool:
    diag_neighbors, total_neighbors = 0, 0
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.safe_access(x, y)
            if not valid_color(cell):
                continue

            for dir_ in Direction:
                offset_x, offset_y = dir_.get_offset()
                neighbor = grid.safe_access(x+offset_x, y+offset_y)
                if not valid_color(neighbor):
                    continue

                total_neighbors += 1
                if dir_.is_diagonal():
                    diag_neighbors += 1

    if total_neighbors == 0:
        return False
    return (diag_neighbors/total_neighbors) > 0.5
