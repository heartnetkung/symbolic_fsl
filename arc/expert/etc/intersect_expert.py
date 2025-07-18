from .intersect import Intersect
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from itertools import combinations
from ..util import *
from ...attention import list_shape_representations


class IntersectExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        if state.has_layer:
            return []

        feat_indexes = list_editable_feature_indexes(task.atn)
        y_shapes = get_y_shapes(state, task.atn)
        result = []

        for index1, index2 in combinations(feat_indexes, 2):
            related_shapes = (get_x_col(state, task.atn, index1) +
                              get_x_col(state, task.atn, index2))

            for color in _find_common_colors(related_shapes):
                candidate = Intersect([index1, index2], color)
                produced_shapes = candidate.intersect(state, task.atn)
                if _check_result(produced_shapes, y_shapes):
                    result.append(candidate)
        return result


def _find_common_colors(shapes: list[Shape])->set[int]:
    if len(shapes) < 2:
        return set()
    result = shapes[0]._grid.list_colors()
    for shape in shapes[1:]:
        result &= shape._grid.list_colors()
        if len(result) == 0:
            break
    return result-{NULL_COLOR}


def _check_result(a_shapes: list[Shape], b_shapes: list[Shape])->bool:
    assert len(a_shapes) == len(b_shapes)
    a_values = [list_shape_representations(a)['colorless_transformed_shape']
                for a in a_shapes]
    b_values = [list_shape_representations(b)['colorless_transformed_shape']
                for b in b_shapes]
    return a_values == b_values
