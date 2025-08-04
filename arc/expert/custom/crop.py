from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import pandas as pd
from ..util import *
from copy import deepcopy
import itertools


class Crop(ModelBasedArcAction[CropTask, CropTask]):
    def __init__(self, bound_model: VisualModel, params: GlobalParams)->None:
        self.bound_model = bound_model
        self.params = params

    def perform(self, state: ArcState, task: CropTask)->Optional[ArcState]:
        assert state.out_shapes != None
        new_out_shapes = []
        df = generate_df(state.x)
        bounds = self.bound_model.predict(state.x, df)

        for canvas, bound in zip(get_canvases(state, task), bounds):
            new_grid = canvas.crop(int(bound[0]), int(
                bound[1]), int(bound[2]), int(bound[3]))
            new_out_shapes.append([Unknown(0, 0, new_grid)])

        if isinstance(state, ArcTrainingState):
            return state.update(out_shapes=new_out_shapes,
                                # needed because y is reparsed
                                y_shapes=new_out_shapes)
        return state.update(out_shapes=new_out_shapes)

    def train_models(self, state: ArcTrainingState,
                     task: CropTask)->list[InferenceAction]:
        # check include_corner
        assert isinstance(self.bound_model, MemorizedVModel)

        df = generate_df(state.x)
        models = bound_model_factory(state.x, df, self.bound_model.result, self.params)
        return [Crop(model, self.params) for model in models]


def get_canvases(state: ArcState, task: CropTask)->list[Grid]:
    if task.crop_from_out_shapes:
        assert state.out_shapes is not None
        assert state.x_bg is not None
        return [draw_canvas(grid.width, grid.height, shapes, bg)
                for grid, shapes, bg in zip(state.x, state.out_shapes, state.x_bg)]
    return state.x
