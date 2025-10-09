from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from .select_one import *
from ..util import *
from collections import Counter


class SelectOneExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None
        # works only on empty attention
        if len(task.atn.x_index[0]) != 0:
            return []

        y_shapes = _get_y_shapes(state.y_shapes)
        if y_shapes is None:
            return []

        result = []
        for grouping in Grouping:
            all_groups, _, sample_index = extract_group(state, grouping)
            label = _make_label(all_groups, sample_index, y_shapes)
            if label is not None:
                result.append(SelectOne(MemorizedModel(label), grouping, self.params))
        return result


def _get_y_shapes(all_y_shapes: list[list[Shape]])->Optional[list[Shape]]:
    result = []
    for y_shapes in all_y_shapes:
        if len(y_shapes) != 1:
            return None
        y_shape = y_shapes[0]
        if (y_shape.x != 0) or (y_shape.y != 0):
            return None
        result.append(y_shape)
    return result


def _make_label(all_groups: list[list[Shape]], sample_index: list[int],
                y_shapes: list[Shape])->Optional[np.ndarray]:
    try:
        result, result_counter = [], Counter()
        for group, sample_id in zip(all_groups, sample_index):
            y_shape = y_shapes[sample_id]
            if group[0]._grid.data == y_shape._grid.data:
                result.append(1)
                result_counter.update([sample_id])
            else:
                result.append(0)

        # assert that all shapes are assigned
        for i in range(len(y_shapes)):
            if result_counter[i] != 1:
                return None
        return np.array(result)
    except Exception:
        return None
