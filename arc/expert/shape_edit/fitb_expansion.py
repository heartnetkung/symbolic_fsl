from ...base import *
from ...graphic import *
from enum import Enum
import numpy as np
from ...algorithm.find_fitb_sym_bound import *


class ExpansionMode(Enum):
    top_left = 0
    top_right = 1
    bottom_left = 2
    bottom_right = 3
    center = 4
    double_mirror = 5
    full_rotation = 6

    def _get_offset(self, diff_width: int, diff_height: int)->Optional[tuple[int, int]]:
        if diff_width < 0 or diff_height < 0:
            return None
        if self == ExpansionMode.top_left:
            return 0, 0
        if self == ExpansionMode.top_right:
            return diff_width, 0
        if self == ExpansionMode.bottom_left:
            return 0, diff_height
        if self == ExpansionMode.bottom_right:
            return diff_width, diff_height
        if self == ExpansionMode.center:
            offset_x, offset_y = diff_width/2, diff_height/2
            if not np.allclose([offset_x % 1, offset_y % 1], 0):
                return None
            return int(offset_x), int(offset_y)
        raise Exception('unknown mode')

    def get_sym_mode(self)->SymmetryMode:
        if self == ExpansionMode.double_mirror:
            return SymmetryMode.double_mirror
        else:
            return SymmetryMode.full_rotation

    def is_symmetry(self)->bool:
        return self in (ExpansionMode.double_mirror, ExpansionMode.full_rotation)

    def get_bound(self, shape: Shape, width: int, height: int)->Optional[
            tuple[int, int, int, int]]:
        if self.is_symmetry():
            if not shape._grid.has_color(NULL_COLOR):
                return None
            bound = find_largest_symmetry(shape._grid)
            if bound is None:
                return None

            sym_mode = self.get_sym_mode()
            blob = cal_offset(shape._grid, bound, sym_mode)
            if blob is None:
                return None

            x, y = blob
            return (x, x+shape.width, y, y+shape.height)

        diff_width, diff_height = width-shape.width, height-shape.height
        offsets = self._get_offset(diff_width, diff_height)
        if offsets is None:
            return None
        return (offsets[0], offsets[0]+shape.width,
                offsets[1], offsets[1]+shape.height)

    def get_widths_heights(self, shapes: list[Shape])->tuple[list[int], list[int]]:
        assert self.is_symmetry()
        widths, heights, sym_mode = [], [], self.get_sym_mode()

        for shape in shapes:
            bound = find_largest_symmetry(shape._grid)
            assert bound is not None
            w, h = cal_width_height(shape._grid, bound, sym_mode)
            widths.append(w)
            heights.append(h)
        return widths, heights
