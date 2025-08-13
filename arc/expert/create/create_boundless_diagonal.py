from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from copy import deepcopy
import pandas as pd
from ..util import *

MAX_GRID_SIZE = 30


class CreateBoundlessDiagonal(
        ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):

    def __init__(self, y_intercept_model: MLModel, color_model: MLModel,
                 orientation_model: MLModel, params: GlobalParams)->None:
        self.y_intercept_model = y_intercept_model
        self.color_model = color_model
        self.orientation_model = orientation_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes != None

        df = default_make_df(state, task)
        y_values = self.y_intercept_model.predict_int(df)
        c_values = self.color_model.predict_int(df)
        o_values = self.orientation_model.predict_bool(df)

        result = deepcopy(state.out_shapes)
        for id1, y_intercept, color, northwest in zip(
                task.atn.sample_index, y_values, c_values, o_values):
            if northwest:
                if y_intercept >= 0:
                    x, y, width = 0, y_intercept, MAX_GRID_SIZE
                else:
                    x, y, width = -y_intercept, 0, MAX_GRID_SIZE
            else:
                x, y, width = 0, 0, y_intercept+1
            result[id1].append(Diagonal(x, y, width, color, northwest))
        return state.update(out_shapes=deduplicate_all_shapes(result))

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.y_intercept_model, MemorizedModel)
        assert isinstance(self.color_model, MemorizedModel)
        assert isinstance(self.orientation_model, MemorizedModel)

        df = default_make_df(state, task)
        y_models = regressor_factory(
            df, self.y_intercept_model.result, self.params, 'bdiag.y')
        c_models = regressor_factory(
            df, self.color_model.result, self.params, 'bdiag.c')
        o_models = classifier_factory(
            df, self.orientation_model.result, self.params, 'bdiag.o')
        return [CreateBoundlessDiagonal(
            y_model, c_model, o_model, self.params)
            for y_model, c_model, o_model in model_selection(
            y_models, c_models, o_models)]
