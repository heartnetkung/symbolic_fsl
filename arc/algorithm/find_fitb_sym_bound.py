from __future__ import annotations
from ..graphic import *
from ..base import *
from typing import Optional
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache


class SymmetryMode(Enum):
    double_mirror = 0
    full_rotation = 1


def cal_width_height(grid: Grid, sym_bound: tuple[int, int, int, int],
                     mode: SymmetryMode)->tuple[int, int]:
    sym_x, sym_y, sym_w, sym_h = sym_bound

    if mode == SymmetryMode.double_mirror:
        return grid.width*2-sym_w, grid.height*2 - sym_h
    elif mode == SymmetryMode.full_rotation:
        bottom_offset, right_offset = grid.height-sym_y-sym_h, grid.width-sym_x-sym_w
        # the largest offset from each edge to the bound
        max_offset = max(sym_x, sym_y, bottom_offset, right_offset)
        return max_offset*2+sym_w, max_offset*2+sym_w
    else:
        raise Exception('unsupported mode')


def cal_offset(grid: Grid, sym_bound: tuple[int, int, int, int],
               mode: SymmetryMode)->Optional[tuple[int, int]]:
    sym_x, sym_y, sym_w, sym_h = sym_bound
    bottom_offset, right_offset = grid.height-sym_y-sym_h, grid.width-sym_x-sym_w

    if mode == SymmetryMode.double_mirror:
        diff_width, diff_height = grid.width-sym_w, grid.height-sym_h
        if (sym_x == 0) and (sym_y == 0):
            return diff_width, diff_height
        elif (sym_x == 0) and (bottom_offset == 0):
            return diff_width, 0
        elif (right_offset == 0) and (sym_y == 0):
            return 0, diff_height
        elif (right_offset == 0) and (bottom_offset == 0):
            return 0, 0
        else:
            return None
    elif mode == SymmetryMode.full_rotation:
        max_offset = max(sym_x, sym_y, bottom_offset, right_offset)
        return max_offset-sym_x, max_offset-sym_y
    else:
        raise Exception('unsupported mode')


@lru_cache
def find_largest_symmetry(grid: Grid)->Optional[tuple[int, int, int, int]]:
    normalized = grid.normalize_color()
    max_size = min(grid.width, grid.height)
    for size in range(max_size-1, 1, -1):
        x_diff, y_diff = grid.width-size, grid.height-size
        for offset_x in range(x_diff+1):
            for offset_y in range(y_diff+1):
                sym = normalized.crop(offset_x, offset_y, size, size)
                if _check_symmetry(sym):
                    return (offset_x, offset_y, size, size)
    return None


def _check_symmetry(grid: Grid)->bool:
    most_common = grid.color_count.most_common(1)
    if len(most_common) == 0:  # empty grid
        return False
    freq, total = most_common[0][1], grid.width*grid.height
    if freq/total < 0.5:
        return False
    return grid == grid.flip_h() == grid.flip_v() == grid.flip_both()
