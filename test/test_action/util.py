from ...arc.base import *
from ...arc.ml import *
from ...arc.graphic import *
from ...arc.expert.recruiter import *
from ...arc.manager import *
import numpy as np
import pandas as pd

DEFAULT_GRID_SIZE = 10
BG = 0


class AttentionExpertProgram(Program[ArcTrainingState]):
    def __init__(self, action: Action, params: GlobalParams,
                 attention_index: int = 0)->None:
        self.action = action
        self.manager = ArcManager(params)
        self.attention_index = attention_index

    def run(self, state: ArcTrainingState)->Optional[ArcTrainingState]:
        task_states = self.manager.decide(state)
        assert len(task_states) > self.attention_index
        task, local_state = task_states[self.attention_index]
        print(local_state,task)
        return self.action.perform(local_state, task)


def create_test_state(x_shapes: list[list[Shape]],
                      y_shapes: list[list[Shape]])->ArcTrainingState:
    x, y, bg = _to_canvas(x_shapes), _to_canvas(y_shapes), [BG]*len(x_shapes)
    return ArcTrainingState(
        x, y, None, bg, bg, False, 5, x_shapes, y_shapes, x_shapes,
        True, True, True, True)


def _to_canvas(all_shapes: list[list[Shape]])->list[Grid]:
    result = []
    for shapes in all_shapes:
        canvas = make_grid(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, BG)
        for shape in shapes:
            shape.draw(canvas)
        result.append(canvas)
    return result
