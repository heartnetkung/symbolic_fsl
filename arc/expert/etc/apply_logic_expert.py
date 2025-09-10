from .apply_logic import ApplyLogic
from .apply_union import ApplyUnion
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from itertools import combinations
from ..util import *
from ...attention import list_shape_representations
from itertools import permutations, chain
from collections import Counter


class ApplyLogicExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
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

            for color in find_common_colors(related_shapes):
                for type in LogicType:
                    candidate = ApplyLogic([index1, index2], color, type)
                    produced_shapes = candidate.apply(state, task.atn)
                    if produced_shapes is None:
                        continue
                    if _check_result(produced_shapes, y_shapes):
                        result.append(candidate)

        indexes = _find_largest_indexes_with_same_size(feat_indexes, state, task.atn)
        if indexes is None:
            return result

        for perm in permutations(indexes):
            candidate = ApplyUnion(list(perm))
            produced_shapes = candidate.apply(state, task.atn)
            if produced_shapes is None:
                continue
            if not _check_result(produced_shapes, y_shapes):
                continue

            result.append(candidate)
        return result


def _check_result(a_shapes: list[Shape], b_shapes: list[Shape])->bool:
    assert len(a_shapes) == len(b_shapes)
    a_values = [list_shape_representations(a)['colorless_transformed_shape']
                for a in a_shapes]
    b_values = [list_shape_representations(b)['colorless_transformed_shape']
                for b in b_shapes]
    return a_values == b_values


def _find_largest_indexes_with_same_size(
        feat_indexes: list[int], state: ArcTrainingState,
        atn: TrainingAttention)->Optional[list[int]]:
    if len(feat_indexes) <= 1:
        return None

    group = {}
    for feat_index in feat_indexes:
        sizes = tuple(((shape.width, shape.height)
                       for shape in get_x_col(state, atn, feat_index)))
        values = group.get(sizes, None)
        if values is None:
            group[sizes] = [feat_index]
        else:
            values.append(feat_index)

    max_value = max(group.values(), key=lambda x: len(x))
    if len(max_value) == 1:
        return None
    return max_value
