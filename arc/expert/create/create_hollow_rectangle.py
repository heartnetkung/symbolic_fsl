from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from ...attention import *
from copy import deepcopy
import pandas as pd
from ..util import *


class CreateHollowRectangle(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, x_model: MLModel, y_model: MLModel, width_model: MLModel,
                 height_model: MLModel, color_model: MLModel, stroke_model: MLModel,
                 params: GlobalParams)->None:
        self.x_model = x_model
        self.y_model = y_model
        self.w_model = width_model
        self.h_model = height_model
        self.c_model = color_model
        self.s_model = stroke_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes != None

        df = default_make_df(state, task)
        x_values = self.x_model.predict_int(df)
        y_values = self.y_model.predict_int(df)
        w_values = self.w_model.predict_int(df)
        h_values = self.h_model.predict_int(df)
        c_values = self.c_model.predict_int(df)
        s_values = self.s_model.predict_int(df)

        result = deepcopy(state.out_shapes)
        for id1, x, y, w, h, c, s in zip(
                task.atn.sample_index, x_values, y_values, w_values, h_values,
                c_values, s_values):
            result[id1].append(HollowRectangle(x, y, w, h, c, s))
        return state.update(out_shapes=deduplicate_all_shapes(result))

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.x_model, MemorizedModel)
        assert isinstance(self.y_model, MemorizedModel)
        assert isinstance(self.w_model, MemorizedModel)
        assert isinstance(self.h_model, MemorizedModel)
        assert isinstance(self.c_model, MemorizedModel)
        assert isinstance(self.s_model, MemorizedModel)

        df = default_make_df(state, task)
        x_models = regressor_factory(df, self.x_model.result, self.params, 'hrect.x')
        y_models = regressor_factory(df, self.y_model.result, self.params, 'hrect.y')
        w_models = regressor_factory(df, self.w_model.result, self.params, 'hrect.w')
        h_models = regressor_factory(df, self.h_model.result, self.params, 'hrect.h')
        c_models = regressor_factory(df, self.c_model.result, self.params, 'hrect.c')
        s_models = regressor_factory(df, self.s_model.result, self.params, 'hrect.s')
        return [CreateHollowRectangle(
            x_model, y_model, w_model, h_model, c_model, s_model, self.params)
            for x_model, y_model, w_model, h_model, c_model, s_model in model_selection(
            x_models, y_models, w_models, h_models, c_models, s_models)]
