from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from .cleanup import CleanUp
from ..util import *


class CleanUpExpert(Expert[ArcTrainingState, CleanUpTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState, task: CleanUpTask)->list[Action]:
        assert state.y_shapes is not None
        assert state.out_shapes is not None

        label = _create_label(state.out_shapes, state.y_shapes)
        model = MemorizedModel(label)
        return [CleanUp(model, self.params)]


def _create_label(all_x_shapes: list[list[Shape]],
                  all_y_shapes: list[list[Shape]])->np.ndarray:
    result = []
    for x_shapes, y_shapes in zip(all_x_shapes, all_y_shapes):
        y_shape_set = set(y_shapes)
        for x_shape in x_shapes:
            result.append(x_shape in y_shape_set)
    return np.array(result)
