from .move_until import MoveUntil, MoveType, UntilType
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from itertools import combinations
from ..util import *


class MoveUntilExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        assert state.out_shapes is not None

        y_shapes = get_y_shapes(state, task.atn)
        all_columns = set(range(len(task.atn.x_index[0])))
        result, atn = [], task.atn

        for moving_index in list_editable_feature_indexes(atn):
            if not has_relationship(atn, 'same_shape', moving_index):
                continue
            if (has_relationship(atn, 'same_x', moving_index) and
                    has_relationship(atn, 'same_y', moving_index)):
                continue

            other_columns = all_columns-{moving_index}
            until_type_blob = _find_until_type(task, other_columns)
            if until_type_blob is None:
                continue

            until_type, until_index = until_type_blob
            for move_type in MoveType:
                candidate = MoveUntil(move_type, until_type, moving_index, until_index)
                new_shapes_blob = candidate.move_all(state, task)
                if new_shapes_blob is None:
                    continue
                if y_shapes != new_shapes_blob[0]:
                    continue

                result.append(candidate)
        return result


def _find_until_type(task: TrainingAttentionTask,
                     columns: set[int])->Optional[tuple[UntilType, int]]:
    for col in columns:
        if has_relationship(task.atn, 'touch', col):
            return UntilType.touch, col
        if has_relationship(task.atn, 'overlap', col):
            return UntilType.overlap, col
    return None
