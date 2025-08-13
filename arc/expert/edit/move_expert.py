from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from .move import Move
from ..util import *


class MoveExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        assert state.y_shapes is not None
        assert state.out_shapes is not None

        result = []
        x_values, y_values = extract_labels(get_y_shapes(state, task.atn))

        for i in list_editable_feature_indexes(task.atn):
            x_shapes = get_x_col(state, task.atn, i)
            current_x_values, current_y_values = extract_labels(x_shapes)
            if (np.array_equal(current_x_values, x_values) and
                    np.array_equal(current_y_values, y_values)):
                continue
            if not has_relationship(task.atn, 'same_shape', i):
                continue

            result.append(Move(MemorizedModel(x_values), MemorizedModel(y_values),
                               self.params, i))
        return result


def extract_labels(shapes: list[Shape])->tuple[np.ndarray, np.ndarray]:
    return (np.array([shape.x for shape in shapes]),
            np.array([shape.y for shape in shapes]))
