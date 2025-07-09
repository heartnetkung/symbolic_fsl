from .util import *
from copy import deepcopy
from typing import Optional
from ..constant import NULL_COLOR


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


def find_tiles(grid: Grid)->Optional[Grid]:
    template1 = _find_row_tiles(grid)
    if template1 is None:
        template2 = _find_row_tiles(grid.transpose())
    else:
        template2 = _find_row_tiles(template1.transpose())

    if template1 is None and template2 is None:  # no tiles
        return None
    elif template2 is None:  # tiles in y direction
        output = template1
    else:  # tiles in x direction or both x and y direction
        output = template2.transpose()

    # incomplete template
    if output is None or NULL_COLOR in output.list_colors():
        return None
    return output


def _find_row_tiles(grid: Grid)->Optional[Grid]:
    if grid.height < 2:
        return None

    for template_height in range(2, grid.height-1):
        template = make_grid(grid.width, template_height)

        # first run to completely fill the template
        for grid2_offset in range(0, grid.height, template_height):
            grid2 = Grid(grid.data[grid2_offset:grid2_offset+template_height])
            template = _match_fill(template, grid2)
            if template is None:
                break

        if template is None:
            continue

        # second run to fully validate it
        for grid2_offset in range(0, grid.height-template_height, template_height):
            grid2 = Grid(grid.data[grid2_offset:grid2_offset+template_height])
            template = _match_fill(template, grid2)
            if template is None:
                break

        if template is not None:
            return template
    return None


def _match_fill(grid1: Grid, grid2: Grid)->Optional[Grid]:
    '''
    Match the color of grid1 and grid2 if they aren't null.
    If a cell in grid1 is null, fill it with the color from grid2.

    Note that grid2 may have less rows
    '''
    assert grid1.width == grid2.width

    for i in range(grid2.height):
        for j in range(grid1.width):
            cell1, cell2 = grid1.data[i][j], grid2.data[i][j]
            if cell1 == NULL_COLOR:
                grid1.data[i][j] = cell2
            elif cell2 != NULL_COLOR:
                if cell1 != cell2:
                    return None

    return grid1


def find_symmetry(grid: Grid, separator: bool)->tuple[Optional[Grid], bool, bool]:
    template1 = _find_row_symmetry(grid, separator)
    if template1 is None:
        template2 = _find_row_symmetry(grid.transpose(), separator)
    else:
        template2 = _find_row_symmetry(template1.transpose(), separator)

    if template1 is None and template2 is None:  # no symmetry
        return None, False, False
    elif template2 is None:  # symmetry in y direction
        output = template1
    else:  # symmetry in x direction or both x and y direction
        output = template2.transpose()

    # incomplete template
    if output is None or NULL_COLOR in output.list_colors():
        output = None
    return output, template1 is not None, template2 is not None


def _find_row_symmetry(grid: Grid, separator: bool)->Optional[Grid]:
    if separator and grid.height < 5:
        return None
    if not separator and grid.height < 4:
        return None
    for template_height in range(2, grid.height-2):
        pattern = grid.crop(0, 0, grid.width, template_height)
        mirror_pattern, pattern_end = _make_mirror_pattern(
            grid, template_height, separator)
        matched_pattern = _match_fill(pattern, mirror_pattern)
        if matched_pattern is not None:
            beyond_pattern = Grid(grid.data[pattern_end+1:]).list_colors()
            if beyond_pattern == set() or beyond_pattern == {NULL_COLOR}:
                return matched_pattern
    return None


def _make_mirror_pattern(grid: Grid, height: int,
                         separator: bool)->tuple[Grid, int]:
    offset = height-1 if separator else height
    grid_width, grid_height = grid.width, grid.height
    output = []
    for i in range(offset+height-1, offset-1, -1):
        if i >= grid_height:
            output.append([NULL_COLOR]*grid_width)
        else:
            output.append(grid.data[i])
    return Grid(output), offset+height-1
