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
        label = _make_label(get_canvases(state, task), state.y)
        if label is None:
            return []

        return [Crop(MemorizedVModel(label), self.params)]


def _make_label(x_grids: list[Grid], y_grids: list[Grid])->Optional[np.ndarray]:
    result = []
    for x_grid, y_grid in zip(x_grids, y_grids):
        offsets = x_grid.find_subgrid(y_grid)
        if offsets is None:
            return None

        result.append([offsets[0], offsets[1], y_grid.width, y_grid.height])
    return np.array(result)
