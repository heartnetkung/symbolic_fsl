from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import FreeDrawTask
import numpy as np
import pandas as pd
from ..util import *
from enum import Enum
from .free_draw import FreeDraw, FreeDrawParam, generate_size_df

DUMMY_MODEL = ConstantModel(1)


class FreeDrawExpert(Expert[ArcTrainingState, FreeDrawTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState, task: FreeDrawTask)->list[Action]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None
        result: list[Action] = [FreeDraw(FreeDrawParam.skip, DUMMY_MODEL, DUMMY_MODEL,
                                         DUMMY_MODEL, self.params)]
        if not _has_exactly_one_shape(state):
            return result

        # since free draw can run on any condition, it will run twice on
        # different unknown_bg. Thus, we stop the second run to save compute.
        if not _is_unknown_bg_false(state):
            return result

        df = generate_size_df(state.x, state.out_shapes)
        if df is None:
            return result

        widths, heights, pixels = _make_label(state.y_shapes)
        blob = _create_wh_models(state.x, widths, heights, df, self.params)
        if blob is None:
            return result

        p_model = StepMemoryModel(pixels)
        result.append(FreeDraw(
            FreeDrawParam.normal, blob[0], blob[1], p_model, self.params))
        return result


def _create_wh_models(
        grids: list[Grid], widths: np.ndarray, heights: np.ndarray,
        df: pd.DataFrame, params: GlobalParams)->Optional[tuple[MLModel, MLModel]]:
    x_widths = [grid.width for grid in grids]
    x_heights = [grid.height for grid in grids]
    are_same = (np.array_equal(x_widths, widths) and
                np.array_equal(x_heights, heights))
    are_constants = len(set(widths)) == len(set(heights)) == 1

    if are_constants:
        return ConstantModel(widths[0]), ConstantModel(heights[0])
    elif are_same:
        return ColumnModel('grid_width'), ColumnModel('grid_height')

    w_models = regressor_factory(df, widths, params, 'fdraw.w')
    h_models = regressor_factory(df, heights, params, 'fdraw.h')
    if (len(w_models) == 0) or (len(h_models) == 0):
        return None
    return w_models[0], h_models[0]


def _make_label(all_shapes: list[list[Shape]])->tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    p_label = []
    for shapes in all_shapes:
        grid = shapes[0]._grid
        for i in range(grid.height):
            for j in range(grid.width):
                p_label.append(grid.data[i][j])

    return (np.array([shapes[0].width for shapes in all_shapes]),
            np.array([shapes[0].height for shapes in all_shapes]),
            np.array(p_label))


def _has_exactly_one_shape(state: ArcTrainingState)->bool:
    assert state.out_shapes is not None
    assert state.y_shapes is not None
    for x_shapes, y_shapes in zip(state.out_shapes, state.y_shapes):
        if (len(x_shapes) != 1) or (len(y_shapes) != 1):
            return False
    return True


def _is_unknown_bg_false(state: ArcTrainingState)->bool:
    assert state.out_shapes is not None
    assert state.x_bg is not None
    assert state.y_bg is not None

    if state.x_bg != state.y_bg:
        return True

    for x_shapes, bg in zip(state.out_shapes, state.x_bg):
        x_shape = x_shapes[0]
        if x_shape._grid.has_color(NULL_COLOR):
            return True
        if x_shape._grid.has_color(bg):
            return False
    return True
