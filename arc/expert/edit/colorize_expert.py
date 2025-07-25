from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from .colorize import Colorize
from ..util import *
from ...manager.reparse.list_reparse_relationship import find_subshape


class ColorizeExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:

        y_shapes = get_y_shapes(state, task.atn)
        label = extract_label(y_shapes)
        if label is None:
            return []

        result = []
        for i in list_editable_feature_indexes(task.atn):
            x_shapes = get_x_col(state, task.atn, i)
            current_values = extract_label(x_shapes)
            if current_values is None:
                continue
            if np.array_equal(current_values, label):
                continue
            if not _is_colorless_subshapes(x_shapes, y_shapes):
                continue
            result.append(Colorize(MemorizedModel(label), self.params, i))
        return result


def extract_label(shapes: list[Shape])->Optional[np.ndarray]:
    result = []
    for shape in shapes:
        if isinstance(shape, FilledRectangle):
            result.append(shape.color)
        elif isinstance(shape, HollowRectangle):
            result.append(shape.color)
        elif isinstance(shape, Diagonal):
            result.append(shape.color)
        elif isinstance(shape, Unknown):
            colors = shape.grid.list_colors()-{NULL_COLOR}
            if len(colors) != 1:
                return None
            result.append(colors.pop())
        else:
            raise Exception('unknown shape implementation')
    return np.array(result)


def _is_colorless_subshapes(x_shapes: list[Shape], y_shapes: list[Shape])->bool:
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        if not _is_colorless_subshape(x_shape, y_shape):
            return False
    return True


def _is_colorless_subshape(x_shape: Shape, y_shape: Shape)->bool:
    if 'colorless_subshape' in find_subshape(x_shape, y_shape, True):
        return True
    if 'colorless_subshape' in find_subshape(y_shape, x_shape, True):
        return True
    return False
