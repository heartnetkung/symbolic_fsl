from .fitp import FITP, find_bound
from .fitb import FillInTheBlank, ExpansionMode
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from ..util import *


class FITPExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
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
            valid_colors = _find_rectangle_colors(x_shapes, y_shapes)
            if valid_colors is None:
                continue

            for color in valid_colors:
                pixels = _extract_pixels(y_shapes)
                color_np = np.array([color]*len(x_shapes))
                width_np = np.array([shape.width for shape in x_shapes])
                height_np = np.array([shape.height for shape in x_shapes])
                subaction = FillInTheBlank(
                    ExpansionMode.top_left, i, MemorizedModel(width_np),
                    MemorizedModel(height_np), StepMemoryModel(pixels), self.params)
                result.append(FITP(subaction, MemorizedModel(color_np)))
        return result


def _find_rectangle_colors(
        x_shapes: list[Shape], y_shapes: list[Shape])->Optional[list[int]]:
    result = []
    for color in range(10):
        if _verify_color(x_shapes, y_shapes, color):
            result.append(color)
    return result


def _verify_color(x_shapes: list[Shape], y_shapes: list[Shape], color: int)->bool:
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        bound = find_bound(x_shape._grid, color)
        if bound is None:
            return False

        x, y, width, height = bound
        if (y_shape.width != width) or (y_shape.height != height):
            return False
    return True


def _extract_pixels(shapes: list[Shape])->np.ndarray:
    result = []
    for shape in shapes:
        for i in range(shape.height):
            for j in range(shape.width):
                result.append(shape._grid.data[i][j])
    return np.array(result)
