from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from .draw_canvas import DrawCanvas, create_df
import numpy as np

DYNAMIC_WIDTH = ColumnModel('bound_width(shapes)')
DYNAMIC_HEIGHT = ColumnModel('bound_height(shapes)')


class DrawCanvasExpert(Expert[ArcTrainingState, DrawCanvasTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params
        self.cache_calculated = False
        self.cached_action: Optional[DrawCanvas] = None

    def solve_problem(self, state: ArcTrainingState,
                      task: DrawCanvasTask)->list[Action]:
        static_action = self._get_static_action(state)
        if static_action is not None:
            return [static_action]

        assert state.out_shapes is not None
        y_width, y_height = _create_labels(state.y)
        df = create_df(state.x, state.out_shapes)
        are_dynamic = (np.allclose(DYNAMIC_WIDTH.predict(df), y_width) and
                       np.allclose(DYNAMIC_HEIGHT.predict(df), y_height))
        if are_dynamic:
            return [DrawCanvas(DYNAMIC_WIDTH, DYNAMIC_HEIGHT)]

        width_models = regressor_factory(df, y_width, self.params, 'draw_all_w')
        height_models = regressor_factory(df, y_height, self.params, 'draw_all_h')
        return [DrawCanvas(width_model, height_model)
                for width_model, height_model in model_selection(
                    width_models, height_models)]

    def _get_static_action(self, state: ArcTrainingState)->Optional[DrawCanvas]:
        if not self.cache_calculated:
            assert state.x is not None
            assert state.y is not None
            assert state.out is not None

            x_width, x_height = _create_labels(state.x)
            y_width, y_height = _create_labels(state.y)
            are_same = (np.array_equal(x_width, y_width) and
                        np.array_equal(x_height, y_height))
            are_constants = len(set(y_width)) == len(set(y_height)) == 1

            if are_constants:
                self.cached_action = DrawCanvas(ConstantModel(y_width[0]),
                                                ConstantModel(y_height[0]))
            elif are_same:
                self.cached_action = DrawCanvas(ColumnModel('grid_width'),
                                                ColumnModel('grid_height'))
            self.cache_calculated = True
        return self.cached_action


def _create_labels(grids: list[Grid])->tuple[np.ndarray, np.ndarray]:
    return (np.array([grid.width for grid in grids]),
            np.array([grid.height for grid in grids]))
