from ...base import *
from ...graphic import *
from ...manager.task import ReparseStackTask
from ...algorithm.solve_stack import solve_stack
from enum import Enum


class ReparseStackParam(Enum):
    skip = 0
    reluctant = 1
    greedy = 2


class ReparseStack(ModelFreeArcAction[ReparseStackTask]):
    '''
    Look at each y_shape and try to break it down into
    multiple shapes stacked on top of each other.
    '''

    def __init__(self, param: ReparseStackParam = ReparseStackParam.skip)->None:
        self.param = param
        super().__init__()

    def perform(self, state: ArcState, task: ReparseStackTask)->Optional[ArcState]:
        assert state.out_shapes is not None
        assert state.x_shapes is not None
        assert state.x_bg is not None
        if self.param == ReparseStackParam.skip:
            return state

        found, new_out_shapes = self._perform_side(
            state.out_shapes, state.x, state.x_bg)
        if not isinstance(state, ArcTrainingState):
            return state.update(out_shapes=new_out_shapes, x_shapes=new_out_shapes,
                                has_layer=True)

        assert state.y_shapes is not None
        assert state.y_bg is not None
        found2, new_y_shapes = self._perform_side(state.y_shapes, state.y, state.y_bg)
        # during training, at least one stack must occur
        if (not found) and (not found2):
            return None

        return state.update(
            out_shapes=new_out_shapes, x_shapes=new_out_shapes, y_shapes=new_y_shapes,
            has_layer=True, reparse_count=state.reparse_count+1)

    def _perform_side(self, all_shapes: list[list[Shape]], grids: list[Grid],
                      bgs: list[int])->tuple[bool, list[list[Shape]]]:
        all_results, found = [], False
        is_greedy = self.param == ReparseStackParam.greedy
        for shapes, grid, bg in zip(all_shapes, grids, bgs):
            grid2 = grid.replace_color(bg, NULL_COLOR)
            stacked_shapes = solve_stack(shapes, grid2, is_greedy)
            sample_results = shapes if stacked_shapes is None else stacked_shapes
            found = True if stacked_shapes is not None else found
            all_results.append(sample_results)

        return found, all_results
