from ...base import *
from ...graphic import *
from ...manager.task import ReparseStackTask
from ...algorithm.solve_stack import solve_stack
from enum import Enum


class ReparseStackParam(Enum):
    skip = 0
    normal = 1


class ReparseStack(TrainingOnlyAction[ArcTrainingState, ReparseStackTask]):
    '''
    Look at each y_shape and try to break it down into
    multiple shapes stacked on top of each other.
    '''

    def __init__(self, param: ReparseStackParam = ReparseStackParam.skip)->None:
        self.param = param
        super().__init__()

    def perform_train(self, state: ArcTrainingState,
                      task: ReparseStackTask)->Optional[ArcTrainingState]:
        assert state.y_shapes is not None
        assert state.y_bg is not None
        if self.param == ReparseStackParam.skip:
            return state

        all_results, found = [], False
        for shapes, grid, bg in zip(state.y_shapes, state.y, state.y_bg):
            grid2 = grid.replace_color(bg, NULL_COLOR)  # TODO why?
            stacked_shapes = solve_stack(shapes, grid2)
            sample_results = shapes if stacked_shapes is None else stacked_shapes
            found = True if stacked_shapes is not None else found
            all_results.append(sample_results)

        if not found:
            return None
        return state.update(y_shapes=all_results, has_layer=True,
                            reparse_count=state.reparse_count+1)
