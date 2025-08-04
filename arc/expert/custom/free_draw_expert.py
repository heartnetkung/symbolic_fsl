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
        if not has_exactly_one_shape(state):
            return result

        df = generate_size_df(state.x, state.out_shapes)
        if df is None:
            return result

        widths, heights, pixels = make_label(state.y_shapes)
        w_models = regressor_factory(df, widths, self.params, 'fdraw.w')
        h_models = regressor_factory(df, heights, self.params, 'fdraw.h')
        p_model = StepMemoryModel(pixels)
        result += [
            FreeDraw(FreeDrawParam.normal, w_model, h_model, p_model, self.params)
            for w_model, h_model in model_selection(w_models, h_models)]
        return result


def make_label(all_shapes: list[list[Shape]])->tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_label = []
    for shapes in all_shapes:
        grid = shapes[0]._grid
        for i in range(grid.height):
            for j in range(grid.width):
                p_label.append(grid.data[i][j])

    return (np.array([shapes[0].width for shapes in all_shapes]),
            np.array([shapes[0].height for shapes in all_shapes]),
            np.array(p_label))


def has_exactly_one_shape(state: ArcTrainingState)->bool:
    assert state.out_shapes is not None
    assert state.y_shapes is not None
    for x_shapes, y_shapes in zip(state.out_shapes, state.y_shapes):
        if (len(x_shapes) != 1) or (len(y_shapes) != 1):
            return False
    return True
