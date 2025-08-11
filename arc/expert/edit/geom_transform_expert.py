from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from .geom_transform import GeomTransform, transform_shape, TransformType
from ..util import *


class GeomTransformExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        y_shapes = get_y_shapes(state, task.atn)
        result, atn = [], task.atn
        for i in list_editable_feature_indexes(task.atn):
            if has_relationship(task.atn, 'colorless_shape', i):
                continue
            x_shapes = get_x_col(state, task.atn, i)
            label = extract_label(x_shapes, y_shapes)
            if label is None:
                continue
            if set(label) == {TransformType.normal.value}:
                continue
            result.append(GeomTransform(MemorizedModel(label), self.params, i))
        return result


def extract_label(x_shapes: list[Shape], y_shapes: list[Shape])->Optional[np.ndarray]:
    for type in TransformType:
        if _check_transform(x_shapes, y_shapes, type):
            return np.array([type.value]*len(x_shapes))
    return None


def _check_transform(x_shapes: list[Shape], y_shapes: list[Shape],
                     type: TransformType)->bool:
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        if transform_shape(x_shape, type)._grid != y_shape._grid:
            return False
    return True
