from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from .draw_canvas import DrawCanvas, create_df, make_sort_label
import numpy as np
from itertools import permutations

DYNAMIC_WIDTH = ColumnModel('bound_width(shapes)')
DYNAMIC_HEIGHT = ColumnModel('bound_height(shapes)')
Cache = Optional[list[list[MLModel]]]


class DrawCanvasExpert(Expert[ArcTrainingState, DrawCanvasTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params
        self.cache_calculated = False
        self.cached_models: Cache = None

    def solve_problem(self, state: ArcTrainingState,
                      task: DrawCanvasTask)->list[Action]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None

        cache = self._get_cache(state)
        if cache is not None:
            all_models: Cache = cache
        else:
            y_width, y_height = _create_labels(state.y)
            df = create_df(state.x, state.out_shapes)
            are_dynamic = (np.allclose(DYNAMIC_WIDTH.predict(df), y_width) and
                           np.allclose(DYNAMIC_HEIGHT.predict(df), y_height))
            if are_dynamic:
                all_models: Cache = [[DYNAMIC_WIDTH], [DYNAMIC_HEIGHT]]
            else:
                labels = [y_width, y_height]
                label_types = [LabelType.reg]*2
                all_models: Cache = make_all_models(
                    df, self.params, 'draw_canvas', labels, label_types)

        if not state.has_layer:
            return [DrawCanvas(w_model, h_model, self.params)
                    for w_model, h_model in model_selection(*all_models)]

        l_models: list[MLModel] = [MemorizedModel(
            _create_sort_label(state.y, state.y_shapes))]
        all_models.append(l_models)
        return [DrawCanvas(w_model, h_model, self.params, l_model)
                for w_model, h_model, l_model in model_selection(*all_models)]

    def _get_cache(self, state: ArcTrainingState)->Cache:
        if not self.cache_calculated:
            assert state.x is not None
            assert state.y is not None

            x_width, x_height = _create_labels(state.x)
            y_width, y_height = _create_labels(state.y)
            are_same = (np.array_equal(x_width, y_width) and
                        np.array_equal(x_height, y_height))
            are_constants = len(set(y_width)) == len(set(y_height)) == 1

            if are_constants:
                self.cached_models = [
                    [ConstantModel(y_width[0])], [ConstantModel(y_height[0])]]
            elif are_same:
                self.cached_models = [
                    [ColumnModel('grid_width')], [ColumnModel('grid_height')]]
            self.cache_calculated = True
        return self.cached_models


def _create_labels(grids: list[Grid])->tuple[np.ndarray, np.ndarray]:
    return (np.array([grid.width for grid in grids]),
            np.array([grid.height for grid in grids]))


def _create_sort_label(grids: list[Grid], all_shapes: list[list[Shape]])->np.ndarray:
    labels = []
    for grid, shapes in zip(grids, all_shapes):
        for shape1, shape2 in permutations(shapes, 2):
            label = make_sort_label(grid, shape1, shape2)
            if label is None:
                continue

            labels.append(not label)
            labels.append(label)
    return np.array(labels)
