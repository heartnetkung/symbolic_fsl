from .hamming import Hamming
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from ..util import *

MIN_SIZE = 2
SIMILARITY_THRESHOLD = 0.8


class HammingExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        assert state.y_shapes is not None
        assert state.out_shapes is not None

        atn = task.atn
        y_shapes = get_y_shapes(state, atn)

        result = []
        for i in list_editable_feature_indexes(atn):
            x_shapes = get_x_col(state, atn, i)
            pixels = _extract_pixels(x_shapes, y_shapes)
            if pixels is None:
                continue

            result.append(Hamming(i, StepMemoryModel(pixels), self.params))
        return result


def _extract_pixels(x_shapes: list[Shape], y_shapes: list[Shape])->Optional[np.ndarray]:
    result = []
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        if (x_shape.width < MIN_SIZE) or (x_shape.height < MIN_SIZE):
            return None
        if (y_shape.width < MIN_SIZE) or (y_shape.height < MIN_SIZE):
            return None

        x_grid, y_grid = x_shape._grid, y_shape._grid
        offsets = _auto_align(x_grid, y_grid)
        if offsets is None:
            return None

        offset_x, offset_y = offsets
        for y in range(x_grid.height):
            for x in range(x_grid.width):
                x_cell = x_grid.safe_access(x, y)
                y_cell = y_grid.safe_access(x-offset_x, y-offset_y)
                if not valid_color(x_cell):
                    if valid_color(y_cell):
                        return None
                    continue

                result.append(max(-1, y_cell))
    return np.array(result)


def _auto_align(x_grid: Grid, y_grid: Grid)->Optional[tuple[int, int]]:
    width_diff = x_grid.width - y_grid.width
    height_diff = x_grid.height - y_grid.height
    if (width_diff < 0) or (height_diff < 0):
        return None

    for offset_y in range(height_diff+1):
        for offset_x in range(width_diff+1):
            if x_grid.offset_subshape_approx(
                    y_grid, offset_x, offset_y, SIMILARITY_THRESHOLD) > 0:
                return offset_x, offset_y
    return None
