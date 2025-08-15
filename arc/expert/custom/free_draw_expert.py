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
        # since free draw can run on any condition, it will run only once per size
        self.size_deduplicator = Deduplicator()

    def solve_problem(self, state: ArcTrainingState, task: FreeDrawTask)->list[Action]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None
        result: list[Action] = [FreeDraw(FreeDrawParam.skip, DUMMY_MODEL, DUMMY_MODEL,
                                         DUMMY_MODEL, self.params)]
        if not _has_exactly_one_shape(state):
            return result

        dedup_key = repr([(
            shapes[0].width, shapes[0].height, shapes[0]._grid.has_color(NULL_COLOR))
            for shapes in state.out_shapes+state.y_shapes])
        if self.size_deduplicator.has_seen_before(dedup_key):
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

    labels = [widths, heights]
    label_types = [LabelType.reg]*2
    all_models = make_all_models(df, params, 'fdraw.size', labels, label_types)
    if (len(all_models[0]) == 0):
        return None
    return all_models[0][0], all_models[1][0]


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
