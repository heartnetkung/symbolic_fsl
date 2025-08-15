from ...base import *
from ...graphic import *
from ...algorithm.recursive_shape_split import recursive_shape_split
from ...algorithm.split_unknown_stack import split_unknown_stack
from enum import Enum
from ...manager.task import ReparseSplitTask
from ...algorithm.find_shapes import *


class ReparseSplitParam(Enum):
    skip = 0
    normal = 1
    colorless = 2
    transform = 3
    approximate = 4


class ReparseSplit(TrainingOnlyAction[ArcTrainingState, ReparseSplitTask]):
    '''
    Look for y_shapes near the image's edge and check if they are partially visible.
    The source of full shapes come from x_shapes.
    '''

    def __init__(self, param: ReparseSplitParam = ReparseSplitParam.skip)->None:
        self.param = param
        super().__init__()

    def perform_train(self, state: ArcTrainingState,
                      task: ReparseSplitTask)->Optional[ArcTrainingState]:
        assert state.y_shapes is not None
        if self.param == ReparseSplitParam.skip:
            return state

        has_layer = state.has_layer
        if self.param == ReparseSplitParam.approximate:
            has_layer = True

        new_shapes = self._reparse(task, state.y_shapes)
        if new_shapes is None:
            return None
        return state.update(y_shapes=new_shapes, has_layer=has_layer,
                            reparse_count=state.reparse_count+1)

    def _reparse(self, task: ReparseSplitTask,
                 all_shapes: list[list[Shape]])->Optional[list[list[Shape]]]:
        all_results, found = [], False
        is_transform = self.param == ReparseSplitParam.transform
        is_colorless = self.param == ReparseSplitParam.colorless
        is_approximate = self.param == ReparseSplitParam.approximate
        subshapes = task.common_y_shapes

        for shapes in all_shapes:
            new_result = []
            for shape in shapes:

                if is_approximate:
                    result = split_unknown_stack(shape, subshapes)
                else:
                    result = recursive_shape_split(
                        shape, subshapes, is_colorless, is_transform)

                if result is None:
                    new_result.append(shape)
                else:
                    found = True
                    new_result += result
            all_results.append(new_result)
        return all_results if found else None
