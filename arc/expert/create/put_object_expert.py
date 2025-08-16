from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from .put_object import PutObject
from ..util import *
from enum import Enum
from collections.abc import Sequence


class TransformType(Enum):
    normal = 0
    colorless = 1
    transform = 2

    def to_repr(self, shape: Shape)->str:
        if self == TransformType.normal:
            return repr(shape)
        if self == TransformType.colorless:
            return repr(shape._grid.normalize_color())
        if self == TransformType.transform:
            return repr(sorted([
                hash(transformed_grid)
                for transformed_grid in geom_transform_all(shape._grid)]))
        raise Exception('unsupported mode')


class PutObjectExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        y_shapes = get_y_shapes(state, task.atn)
        if not _has_unknown_object(y_shapes):
            return []

        result = []
        # from common y shapes
        blob = _create_labels(task.common_y_shapes, y_shapes)
        if blob is not None:
            selection, x, y = blob
            result.append(PutObject(MemorizedModel(selection), MemorizedModel(x),
                                    MemorizedModel(y), self.params))
        return result


def _has_unknown_object(y_shapes: list[Shape])->bool:
    for y_shape in y_shapes:
        if isinstance(y_shape, Unknown):
            return True
    return False


def _create_labels(common_shapes: Sequence[Shape], y_shapes: list[Shape])->Optional[
        tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for type in TransformType:
        selections = _create_common_shapes_label(common_shapes, y_shapes, type)
        if selections is None:
            continue

        x_values = np.array([shape.x for shape in y_shapes])
        y_values = np.array([shape.y for shape in y_shapes])
        return selections, x_values, y_values
    return None


def _create_common_shapes_label(common_shapes: Sequence[Shape], y_shapes: list[Shape],
                                type: TransformType)->Optional[np.ndarray]:
    mapping = {type.to_repr(common_shape): i
               for i, common_shape in enumerate(common_shapes)}
    if len(mapping) != len(common_shapes):  # repeated representation
        return None

    result = []
    for y_shape in y_shapes:
        key = type.to_repr(y_shape)
        value = mapping.get(key, None)
        if value is None:
            return None
        result.append(value)
    return np.array(result)
