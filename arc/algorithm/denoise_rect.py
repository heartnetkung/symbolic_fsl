from ..graphic import *
import numpy as np
from ..constant import *
from typing import Optional
from collections import Counter

NOISE_MASS = 4
NOISE_RATIO = 0.35


def denoise_rect(grid: Grid, width: int, height: int)->Optional[Grid]:
    '''Denoise algorithm assuming the underlying shapes are single-colored rectangles'''

    if (grid.width != width) or (grid.height != height):
        return None

    blob = find_signal_noise(grid)
    if blob is None:
        return None

    signal, noise = blob
    if len(noise) > 0:
        return _reparse(_replace_noise(grid, noise, signal))
    return _reparse(grid)


def find_signal_noise(grid: Grid)->Optional[tuple[set[int], set[int]]]:
    color_freq = grid.color_count.most_common()
    if len(color_freq) == 0:
        return None
    if len(color_freq) == 1:
        return ({color_freq[0][0]}, set())

    mean_freq = np.mean([freq for color, freq in color_freq])
    signal, noise = set(), set()
    for color, freq in color_freq:
        if freq > mean_freq:
            signal.add(color)
        else:
            noise.add(color)
    return signal, noise


def _replace_noise(grid: Grid, noise: set[int], signal: set[int])->Grid:
    result = make_grid(grid.width, grid.height)
    for i in range(grid.height):
        for j in range(grid.width):
            cell = grid.data[i][j]
            if cell in noise:
                nearbys = [grid.safe_access(j+1, i), grid.safe_access(j-1, i),
                           grid.safe_access(j, i+1), grid.safe_access(j, i-1)]
                nearbys = [val for val in nearbys if val in signal]

                if len(nearbys) < 2:
                    result.data[i][j] = NULL_COLOR
                else:
                    result.data[i][j] = Counter(nearbys).most_common(1)[0][0]
            else:
                result.data[i][j] = cell
    return result


def _reparse(grid: Grid)->Grid:
    shapes = list_objects(grid, True)
    reparsed_shapes = []
    for shape in shapes:
        if shape.width * shape.height <= NOISE_MASS:
            continue
        if isinstance(shape, FilledRectangle):
            reparsed_shapes.append(shape)
        else:
            reparsed_shapes += _to_rect(shape)
    return draw_canvas(grid.width, grid.height, reparsed_shapes)


def _find_first_color(grid: Grid)->int:
    for i in range(grid.height):
        for j in range(grid.width):
            cell = grid.data[i][j]
            if valid_color(cell):
                return cell
    return NULL_COLOR


def _to_rect(shape: Shape)->list[Shape]:
    '''find lowest possible '''
    color = _find_first_color(shape._grid)
    offsets_x, endsets_x = _get_row_info(shape._grid)
    offset_x = _resolve_offset(offsets_x)
    endset_x = _resolve_offset(endsets_x)

    offsets_y, endsets_y = _get_row_info(shape._grid.transpose())
    offset_y = _resolve_offset(offsets_y)
    endset_y = _resolve_offset(endsets_y)

    w, h = shape.width-offset_x-endset_x, shape.height-offset_y-endset_y
    grid2 = shape._grid.crop(offset_x, offset_y, w, h)

    # if the grid is almost rectangle, just return it as rectangle.
    noise_ratio = np.sum(np.array(grid2.data) == NULL_COLOR) / (w*h)
    if noise_ratio < NOISE_RATIO:
        return [FilledRectangle(shape.x+offset_x, shape.y+offset_y, w, h, color)]

    # if the grid is far from complete, keep it as it is.
    return [Unknown(shape.x, shape.y, grid2)]


def _resolve_offset(offsets: list[int])->int:
    '''Get the least offset that pass the noise ratio.'''

    counter, max_val = Counter(offsets), max(offsets)
    for i in range(max_val):
        if counter[i]/counter.total() > NOISE_RATIO:
            return i
    return max_val


def _get_row_info(grid: Grid)->tuple[list[int], list[int]]:
    starts, ends = [], []
    for row in grid.data:
        starts.append(_find_start(row, True))
        ends.append(_find_start(row, False))
    return starts, ends


def _find_start(row: list[int], from_start: bool)->Optional[tuple[int, int]]:
    range_ = range(len(row)) if from_start else range(len(row)-1, -1, -1)
    for i in range_:
        cell = row[i]
        if cell != NULL_COLOR:
            return i if from_start else len(row)-i-1
    return None
