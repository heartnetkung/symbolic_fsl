from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from .colorize import Colorize
from ..util import *


class ColorizeExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        assert state.y_shapes is not None
        assert state.out_shapes is not None

        label = extract_label(get_y_shapes(state, task.atn))
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
            if not has_relationship(task.atn, 'same_colorless_shape', i):
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
            colors = shape.grid.list_colors()
            if len(colors) != 1:
                return None
            result.append(colors.pop())
        else:
            raise Exception('unknown shape implementation')
    return np.array(result)
