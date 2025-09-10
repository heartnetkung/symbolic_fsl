from ...graphic import *
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from ...constant import *
import numpy as np


MAX_ITR = 60


class ShapeConvProcess(ABC):
    '''A processing unit to run on every convolution.'''

    def _postprocess(self, shapes: list[Shape])->list[Shape]:
        return shapes

    def _unit_label(self, y_grid: Grid, offset_x: int, offset_y: int)->list[float]:
        raise Exception('not implemented')

    def _get_x_range(self, grid: Grid)->range:
        return range(grid.width)

    def _get_y_range(self, grid: Grid)->range:
        return range(grid.height)

    def _can_start(self, grid: Grid, offset_x: int, offset_y: int)->bool:
        return True

    def _can_expand(self, grid: Grid, offset_x: int, offset_y: int,
                    w: int, h: int, previous: bool, is_right: bool)->bool:
        return False

    @abstractmethod
    def _to_result(self, grid: Grid, offset_x: int, offset_y: int,
                   w: int, h: int)->Optional[Shape]:
        pass

    def _unit_process(
            self, x_grid: Grid, offset_x: int, offset_y: int)->Optional[Shape]:
        width, height = 1, 1
        previous_expand_right, previous_expand_bottom = True, True
        for _ in range(MAX_ITR):
            expand_right = self._can_expand(x_grid, offset_x, offset_y, width, height,
                                            previous_expand_right, True)
            expand_bottom = self._can_expand(x_grid, offset_x, offset_y, width, height,
                                             previous_expand_bottom, False)

            if expand_right and expand_bottom:
                if (width+1)*height >= (height+1)*width:
                    is_right_selected = True
                else:
                    is_right_selected = False
            elif expand_right:
                is_right_selected = True
                previous_expand_bottom = False
            elif expand_bottom:
                is_right_selected = False
                previous_expand_right = False
            else:
                result = self._to_result(x_grid, offset_x, offset_y, width, height)
                if result is None:
                    continue
                return result

            if is_right_selected:
                width += 1
            else:
                height += 1
        return None

    def process(self, x_grid: Grid)->Optional[list[Shape]]:
        result = NonOverlapingContainer(x_grid.width, x_grid.height)
        for x in self._get_x_range(x_grid):
            for y in self._get_y_range(x_grid):
                if not self._can_start(x_grid, x, y):
                    continue

                new_shape = self._unit_process(x_grid, x, y)
                if new_shape is None:
                    continue

                success = result.add(new_shape)
                if not success:
                    overlaps = result.query_overlap(new_shape)
                    all_areas = [shape.width*shape.height for shape in overlaps]
                    new_shape_area = new_shape.width*new_shape.height
                    if new_shape_area > max(all_areas, default=0):
                        for overlap in overlaps:
                            result.remove(overlap)
                        result.add(new_shape)
        return self._postprocess(list(result.items()))

    def make_label(self, y_grid: Grid)->Optional[list[np.ndarray]]:
        labels = []
        for x in self._get_x_range(y_grid):
            for y in self._get_y_range(y_grid):
                try:
                    if not self._can_start(y_grid, x, y):
                        continue

                    new_rows = self._unit_label(y_grid, x, y)
                    for label, new_row in zip(labels, new_rows):
                        label.append(new_row)
                except Exception:
                    return None
        return [np.array(label) for label in labels]
