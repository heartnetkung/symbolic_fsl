from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from ..util import *
from .draw_rect import DrawRect, SelectCondition
from .draw_intersect import DrawIntersect

ConvDrawAction = Union[DrawRect, DrawIntersect]


class ConvolutionDrawExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        assert state.y_shapes is not None
        assert state.out_shapes is not None

        y_shapes = get_y_shapes(state, task.atn)
        result: list[ConvDrawAction] = []
        for i in list_editable_feature_indexes(task.atn):
            x_shapes = get_x_col(state, task.atn, i)
            labels = _extract_labels(x_shapes, y_shapes)
            if labels is None:
                continue

            old_colors, new_colors = labels
            result += _make_draw_rect(
                x_shapes, y_shapes, old_colors, new_colors, i, self.params)
            result += _make_draw_intersect(
                x_shapes, y_shapes, old_colors, new_colors, i, self.params)

        checked_result = []
        for action in result:
            new_state = action.perform(state, task)  # type:ignore
            if new_state is None:
                continue

            new_out_shapes = get_x_col(new_state, task.atn, action.feat_index)
            if new_out_shapes != y_shapes:
                continue

            checked_result.append(action)
        return checked_result


def _extract_labels(x_shapes: list[Shape], y_shapes: list[Shape])->Optional[
        tuple[list[int], list[int]]]:
    old_colors, new_colors, unchanged_colors = [], [], []
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        if (not isinstance(x_shape, Unknown)) or (not isinstance(y_shape, Unknown)):
            return None
        if (x_shape.width, x_shape.height) != (y_shape.width, y_shape.height):
            return None

        x_grid, y_grid = x_shape.grid, y_shape.grid
        if len(x_grid.list_colors()) < 2 or len(y_grid.list_colors()) < 2:
            return None

        before_colors, after_colors = set(), set()
        for i in range(x_shape.height):
            for j in range(x_shape.width):
                x_cell, y_cell = x_grid.data[i][j], y_grid.data[i][j]
                if x_cell != y_cell:
                    before_colors.add(x_cell)
                    after_colors.add(y_cell)

        if (len(before_colors) != 1) or (len(after_colors) != 1):
            return None

        old_colors.append(before_colors.pop())
        new_colors.append(after_colors.pop())
    return old_colors, new_colors


def _make_draw_rect(x_shapes: list[Shape], y_shapes: list[Shape], old_colors: list[int],
                    new_colors: list[int], feat_index: int,
                    params: GlobalParams)->list[ConvDrawAction]:
    old_color_model = MemorizedModel(np.array(old_colors))
    new_color_model = MemorizedModel(np.array(new_colors))
    has_various_areas = False

    for y_shape, new_color in zip(y_shapes, new_colors):
        new_color_subshapes = [shape for shape in list_objects(y_shape._grid)
                               if shape.single_color == new_color]

        areas = set()
        for subshape in new_color_subshapes:
            if not isinstance(subshape, FilledRectangle):
                return []
            areas.add(subshape.width*subshape.height)

        if len(areas) > 1:
            has_various_areas = True

    selections = [SelectCondition.min_size] if has_various_areas else SelectCondition
    return [DrawRect(selection, feat_index, old_color_model, new_color_model, params)
            for selection in SelectCondition]


def _make_draw_intersect(x_shapes: list[Shape], y_shapes: list[Shape],
                         old_colors: list[int], new_colors: list[int], feat_index: int,
                         params: GlobalParams)->list[ConvDrawAction]:
    old_color_model = MemorizedModel(np.array(old_colors))
    new_color_model = MemorizedModel(np.array(new_colors))

    unchanged_colors = set(range(10))
    for y_shape in y_shapes:
        unchanged_colors &= y_shape._grid.list_colors()
    unchanged_colors -= set(new_colors)

    return [DrawIntersect(feat_index, old_color_model, new_color_model, color, params)
            for color in sorted(unchanged_colors)]
