from .split_shape import SplitShape
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from ..util import *


class SplitShapeExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        n_col = len(task.atn.x_index[0])
        if n_col == 0:
            return []

        result, atn = [], task.atn
        grouping = _group_attentions(atn, state)
        for i in range(n_col):
            if not has_relationship(atn, 'full_overlap', i):
                continue
            if not _check_mass(grouping, i):
                continue
            result.append(SplitShape(i))
        return result


def _check_mass(grouping: list[tuple[list[Shape], list[Shape]]], column: int)->bool:
    for x_shapes, y_shapes in grouping:
        x_shape = x_shapes[column]
        total_mass = sum([y_shape.mass for y_shape in y_shapes])
        if x_shape.mass != total_mass:
            return False
    return True


def _group_attentions(atn: TrainingAttention,
                      state: ArcTrainingState)->list[tuple[list[Shape], list[Shape]]]:
    assert state.y_shapes is not None
    assert state.out_shapes is not None

    mapping: dict[str, tuple[list[Shape], list[Shape]]] = {}
    for id1, x_indexes, y_index in zip(atn.sample_index, atn.x_index, atn.y_index):
        key = repr((id1, x_indexes))
        value = mapping.get(key, None)
        new_y_shapes = [state.y_shapes[id1][y_index]]
        if value is None:
            x_shapes = [state.out_shapes[id1][i] for i in x_indexes]
            mapping[key] = (x_shapes, new_y_shapes)
        else:
            x_shapes, y_shapes = value
            mapping[key] = (x_shapes, y_shapes+new_y_shapes)
    return list(mapping.values())
