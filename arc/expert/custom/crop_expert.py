from .crop import Crop, get_canvases
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from ..util import *


class CropExpert(Expert[ArcTrainingState, CropTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState, task: CropTask)->list[Action]:
        labels = _make_label(get_canvases(state, task), state.y)
        if labels is None:
            return []

        label, label2 = labels
        return [Crop(StepMemoryModel(label), self.params, True),
                Crop(StepMemoryModel(label2), self.params, False)]


def _make_label(x_grids: list[Grid], y_grids: list[Grid])->Optional[
        tuple[np.ndarray, np.ndarray]]:
    result, result2 = [], []
    for x_grid, y_grid in zip(x_grids, y_grids):
        offsets = x_grid.find_subgrid(y_grid)
        if offsets is None:
            return None

        offset_x, offset_y = offsets
        w, h = y_grid.width, y_grid.height
        correct_coords = {(offset_x, offset_y), (offset_x+w-1, offset_y),
                          (offset_x, offset_y+h-1), (offset_x+w-1, offset_y+h-1)}
        current_coords2 = {(offset_x-1, offset_y-1), (offset_x+w, offset_y-1),
                           (offset_x-1, offset_y+h), (offset_x+w, offset_y+h)}

        for x in range(x_grid.width):
            for y in range(x_grid.height):
                result.append(1 if (x, y) in correct_coords else 0)
                result2.append(1 if (x, y) in current_coords2 else 0)

    return np.array(result), np.array(result2)
