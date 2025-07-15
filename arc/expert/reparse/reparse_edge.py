from ...base import *
from ...graphic import *
from ...manager.reparse import *
from ...algorithm.resolve_edge import resolve_edge
from enum import Enum


class ReparseEdgeParam(Enum):
    skip = 0
    normal = 1
    colorless = 2


class ReparseEdge(TrainingOnlyAction[ArcTrainingState, ReparseEdgeTask]):
    '''
    Look for y_shapes near the image's edge and check if they are partially visible.
    The source of full shapes come from x_shapes.
    '''

    def __init__(self, param: ReparseEdgeParam = ReparseEdgeParam.skip)->None:
        self.param = param
        super().__init__()

    def perform_train(self, state: ArcTrainingState,
                      task: ReparseEdgeTask)->Optional[ArcTrainingState]:
        assert state.y_shapes is not None
        if self.param == ReparseEdgeParam.skip:
            return state

        if self.param == ReparseEdgeParam.colorless:
            graph = task.colorless_supershape
        else:
            graph = task.supershape
        if graph.number_of_edges() == 0:
            return None

        new_y_shapes = self._reparse(graph, state.y_shapes, state.y)
        if new_y_shapes is None:
            return None

        return state.update(y_shapes=new_y_shapes, reparse_count=state.reparse_count+1)

    def _reparse(self, graph: ShapeGraph, all_shapes: list[list[Shape]],
                 grids: list[Grid])->Optional[list[list[Shape]]]:
        all_results, found = [], False
        is_colorless = self.param == ReparseEdgeParam.colorless

        for shapes, grid in zip(all_shapes, grids):
            new_result = []
            for shape in shapes:
                subshape_tuples = graph.lookup(shape)
                if len(subshape_tuples) != 1:
                    new_result.append(shape)
                    continue

                subshape = subshape_tuples[0][0]
                result = resolve_edge(shape, subshape, grid, is_colorless)
                if result is None:
                    new_result.append(shape)
                else:
                    found = True
                    new_result.append(result)
            all_results.append(new_result)

        return all_results if found else None
