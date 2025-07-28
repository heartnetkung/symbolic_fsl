from ...base import *
from ...graphic import *
from ...algorithm.recursive_shape_split import recursive_shape_split
from ...algorithm.split_unknown_stack import split_unknown_stack
from enum import Enum
from ...manager.reparse import *


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
        if self.param == ReparseSplitParam.transform:
            graph = task.transformed_subshape
        elif self.param == ReparseSplitParam.approximate:
            graph = task.approx_subshape
            has_layer = True
        else:
            graph = task.subshape
        if graph.number_of_edges() not in range(1, MAX_REPARSE_EDGE):
            return None

        new_shapes = self._reparse(graph, state.y_shapes)
        if new_shapes is None:
            return None
        return state.update(y_shapes=new_shapes, has_layer=has_layer,
                            reparse_count=state.reparse_count+1)

    def _reparse(self, graph: ShapeGraph,
                 all_shapes: list[list[Shape]])->Optional[list[list[Shape]]]:
        all_results, found = [], False
        is_transform = self.param == ReparseSplitParam.transform
        is_colorless = self.param == ReparseSplitParam.colorless
        is_approximate = self.param == ReparseSplitParam.approximate

        for shapes in all_shapes:
            new_result = []
            for shape in shapes:
                subshape_tuples = graph.lookup(shape)
                if len(subshape_tuples) == 0:
                    new_result.append(shape)
                    continue

                subshapes = [shape for shape, data in subshape_tuples]
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
