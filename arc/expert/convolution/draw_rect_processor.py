from ...base import *
from ...graphic import *
from copy import deepcopy
from ..util import *


class SelectCondition(Enum):
    min_size = 0
    largest = 1


class DrawRectProcessor(ShapeConvProcess):
    def __init__(self, old_color: int, new_color: int, selection: SelectCondition,
                 min_width: int, min_height: int)->None:
        self.old_color = old_color
        self.new_color = new_color
        self.selection = selection
        self.min_width = min_width
        self.min_height = min_height

    def _can_start(self, grid: Grid, offset_x: int, offset_y: int)->bool:
        if grid.data[offset_y][offset_x] != self.old_color:
            return False
        return True

    def _to_result(self, grid: Grid, offset_x: int, offset_y: int,
                   w: int, h: int)->Optional[Shape]:
        if (w < self.min_width) or (h < self.min_height):
            return None
        return FilledRectangle(offset_x, offset_y, w, h, self.new_color)

    def _can_expand(self, grid: Grid, offset_x: int, offset_y: int,
                    w: int, h: int, previous: bool, is_right: bool)->bool:
        if previous == False:
            return False
        if is_right and (offset_x+w+1 > grid.width):
            return False
        if (not is_right) and (offset_y+h+1 > grid.height):
            return False

        if is_right:
            to_check = {grid.safe_access(offset_x+w, i)
                        for i in range(offset_y, offset_y+h)}
        else:
            to_check = {grid.safe_access(j, offset_y+h)
                        for j in range(offset_x, offset_x+w)}
        return to_check == {self.old_color}

    def _postprocess(self, shapes: list[Shape])->list[Shape]:
        if self.selection == SelectCondition.largest:
            all_areas = [shape.width*shape.height for shape in shapes]
            largest_area = max(all_areas, default=0)
            result = [shape for shape in shapes
                      if shape.width*shape.height == largest_area]
            return result
        elif self.selection == SelectCondition.min_size:
            return shapes
        else:
            raise Exception('unsupported type')
