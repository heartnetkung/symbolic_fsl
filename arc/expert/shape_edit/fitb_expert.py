from .fitb import ExpansionMode, FillInTheBlank
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from ..util import *


class FilleInTheBlankExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        assert state.y_shapes is not None
        assert state.out_shapes is not None

        atn = task.atn
        y_shapes = get_y_shapes(state, atn)
        widths, heights = _extract_sizes(y_shapes)

        result = []
        for i in list_editable_feature_indexes(atn):
            if not has_relationship(atn, 'subshape', i):
                continue

            x_shapes = get_x_col(state, atn, i)
            expansions = _get_expansions(x_shapes, y_shapes, widths, heights)
            if expansions is None:
                continue

            mode, expanded_shapes = expansions
            pixels = _extract_pixels(x_shapes, y_shapes, widths, heights, mode)
            if pixels is None:
                continue

            result.append(FillInTheBlank(
                mode, i, MemorizedModel(widths), MemorizedModel(heights),
                MemorizedModel(pixels), self.params))
        return result


def _extract_sizes(shapes: list[Shape])->tuple[np.ndarray, np.ndarray]:
    widths = [shape.width for shape in shapes]
    heights = [shape.height for shape in shapes]
    return np.array(widths), np.array(heights)


def _get_expansions(x_shapes: list[Shape], y_shapes: list[Shape], widths: np.ndarray,
                    heights: np.ndarray)->Optional[tuple[ExpansionMode, list[Unknown]]]:
    for mode in ExpansionMode:
        new_shapes = []
        for x_shape, y_shape, width, height in zip(x_shapes, y_shapes, widths, heights):
            new_shape = mode.expand(x_shape, width, height)
            is_equal = nonnull_equal(x_shape._grid, y_shape._grid)
            if new_shape is None or (not is_equal):
                new_shapes = []
                break
            new_shapes.append(new_shape)

        if len(new_shapes) != 0:
            return mode, new_shapes
    return None


def nonnull_equal(a: Grid, b: Grid)->bool:
    if a.width != b.width or a.height != b.height:
        return False

    for i in range(a.height):
        for j in range(a.width):
            cell_a, cell_b = a.data[i][j], b.data[i][j]
            if cell_a == NULL_COLOR or cell_b == NULL_COLOR:
                continue
            if cell_b != cell_a:
                return False
    return True


def _extract_pixels(x_shapes: list[Shape], y_shapes: list[Shape], widths: np.ndarray,
                    heights: np.ndarray, mode: ExpansionMode)->Optional[np.ndarray]:
    result = []
    for x_shape, y_shape, width, height in zip(x_shapes, y_shapes, widths, heights):
        expanded_shape = mode.expand(x_shape, width, height)
        if expanded_shape is None:
            return None

        x_grid, y_grid = expanded_shape._grid, y_shape._grid
        for y in range(x_grid.height):
            for x in range(x_grid.width):
                x_cell = x_grid.safe_access(x, y)
                if x_cell != NULL_COLOR:
                    continue

                y_cell = y_grid.safe_access(x, y)
                if y_cell == MISSING_VALUE:
                    return None

                result.append(y_cell)
    return np.array(result)
