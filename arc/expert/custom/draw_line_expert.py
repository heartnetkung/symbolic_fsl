from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import numpy as np
import pandas as pd
from ..util import *
from enum import Enum
from .draw_line import DrawLine


class DrawLineExpert(Expert[ArcTrainingState, TrainingDrawLineTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: TrainingDrawLineTask)->list[Action]:
        blob = _extract_labels(task.get_attention_aligned_lines())
        if blob is None:
            return []

        x_values, y_values, color_values, dir_values, nav_values = blob
        return [DrawLine(MemorizedModel(x_values), MemorizedModel(y_values),
                         MemorizedModel(dir_values), StepMemoryModel(nav_values),
                         MemorizedModel(color_values), self.params)]


def _extract_labels(lines: list[Line])->Optional[tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    x_values, y_values, color_values, dir_values, nav_values = [], [], [], [], []
    for line in lines:
        start, _ = line.get_start_end_pixels()
        x_values.append(start.x)
        y_values.append(start.y)
        color_values.append(start.color)

        blob = line.to_dir()
        if blob is None:
            return None

        dir_, navs = blob
        dir_values.append(dir_.value)
        nav_values += [nav.value for nav in navs]

    return (np.array(x_values), np.array(y_values), np.array(color_values),
            np.array(dir_values), np.array(nav_values))
