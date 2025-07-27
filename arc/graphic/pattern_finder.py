from .util import *
from typing import Optional
from ..constant import NULL_COLOR
import math


def find_h_symmetry(grid: Grid)->Optional[tuple[int, int]]:
    '''Find minimum height and offset to achive horizontal symmetry.'''
    if grid.height < 4:
        return None

    result: Optional[tuple[int, int]] = None
    count = 0

    for sym_height in range(grid.height, 3, -1):
        for offset in range(grid.height+1-sym_height):
            new_count = _overlap_count(grid, sym_height, offset)
            if new_count > count:
                count = new_count
                result = (sym_height, offset)
    return result


def find_v_symmetry(grid: Grid)->Optional[tuple[int, int]]:
    return find_h_symmetry(grid.transpose())


def _overlap_count(grid: Grid, height: int, offset: int)->int:
    count = 0
    for i in range(math.floor(height/2)):
        row1, row2 = grid.data[i+offset], grid.data[height-1-i+offset]
        for cell1, cell2 in zip(row1, row2):
            if cell1 == cell2:
                count += 1
    return count
