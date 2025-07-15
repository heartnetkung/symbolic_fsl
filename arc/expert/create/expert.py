from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from ..util import *
from .create_diagonal import CreateDiagonal
from .create_rectangle import CreateRectangle
from .create_hollow_rectangle import CreateHollowRectangle
from .create_boundless_diagonal import CreateBoundlessDiagonal


class CreateExpert(Expert[ArcTrainingState, TrainingAttentionTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingAttentionTask)->list[Action]:
        assert state.y_shapes is not None
        assert state.out_shapes is not None

        result, atn, params = [], task.atn, self.params
        y_shapes = get_y_shapes(state, atn)
        df = default_make_df(state, atn)

        rectangles = _to_rect(y_shapes)
        if rectangles is not None:
            models = [MemorizedModel(label)
                      for label in _extract_rect_labels(rectangles)]
            result.append(CreateRectangle(
                models[0], models[1], models[2], models[3], models[4], params))

        h_rectangles = _to_hrect(y_shapes)
        if h_rectangles is not None:
            models = [MemorizedModel(label)
                      for label in _extract_hrect_labels(h_rectangles)]
            result.append(CreateHollowRectangle(
                models[0], models[1], models[2], models[3], models[4],
                models[5], params))

        diags = _to_diag(y_shapes)
        if diags is not None:
            models = [MemorizedModel(label)
                      for label in _extract_diag_labels(diags)]
            result.append(CreateDiagonal(
                models[0], models[1], models[2], models[3], models[4], params))
        return result


def _to_rect(shapes: list[Shape])->Optional[list[FilledRectangle]]:
    result = []
    for shape in shapes:
        if isinstance(shape, FilledRectangle):
            result.append(shape)
        elif (isinstance(shape, Unknown) and
                FilledRectangle.is_valid(shape.x, shape.y, shape.grid)):
            result.append(FilledRectangle.from_grid(shape.x, shape.y, shape.grid))
        else:
            return None
    return result


def _to_hrect(shapes: list[Shape])->Optional[list[HollowRectangle]]:
    result = []
    for shape in shapes:
        if isinstance(shape, HollowRectangle):
            result.append(shape)
        elif (isinstance(shape, Unknown) and
                HollowRectangle.is_valid(shape.x, shape.y, shape.grid)):
            result.append(HollowRectangle.from_grid(shape.x, shape.y, shape.grid))
        else:
            return None
    return result


def _to_diag(shapes: list[Shape])->Optional[list[Diagonal]]:
    result = []
    for shape in shapes:
        if isinstance(shape, Diagonal):
            result.append(shape)
        elif (isinstance(shape, Unknown) and
                Diagonal.is_valid(shape.x, shape.y, shape.grid)):
            result.append(Diagonal.from_grid(shape.x, shape.y, shape.grid))
        else:
            return None
    return result


def _extract_rect_labels(shapes: list[FilledRectangle])->tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([shape.x for shape in shapes]),
            np.array([shape.y for shape in shapes]),
            np.array([shape.width for shape in shapes]),
            np.array([shape.height for shape in shapes]),
            np.array([shape.color for shape in shapes]))


def _extract_hrect_labels(shapes: list[HollowRectangle])->tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([shape.x for shape in shapes]),
            np.array([shape.y for shape in shapes]),
            np.array([shape.width for shape in shapes]),
            np.array([shape.height for shape in shapes]),
            np.array([shape.color for shape in shapes]),
            np.array([shape.stroke for shape in shapes]))


def _extract_diag_labels(shapes: list[Diagonal])->tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([shape.x for shape in shapes]),
            np.array([shape.y for shape in shapes]),
            np.array([shape.width for shape in shapes]),
            np.array([shape.color for shape in shapes]),
            np.array([shape.north_west for shape in shapes]))
