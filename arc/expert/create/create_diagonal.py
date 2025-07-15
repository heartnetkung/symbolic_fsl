from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from ...attention import *
from copy import deepcopy
import pandas as pd
from ..util import *


class CreateDiagonal(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, x_model: MLModel, y_model: MLModel, width_model: MLModel,
                 color_model: MLModel, orientation_model: MLModel,
                 params: GlobalParams)->None:
        self.x_model = x_model
        self.y_model = y_model
        self.w_model = width_model
        self.c_model = color_model
        self.o_model = orientation_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes != None

        atn = task.atn
        df = default_make_df(state, atn)
        x_values = self.x_model.predict_int(df)
        y_values = self.y_model.predict_int(df)
        w_values = self.w_model.predict_int(df)
        c_values = self.c_model.predict_int(df)
        o_values = self.o_model.predict_bool(df)

        result = deepcopy(state.out_shapes)
        for id1, x, y, w, c, o in zip(
                atn.sample_index, x_values, y_values, w_values, c_values, o_values):
            result[id1].append(Diagonal(x, y, w, c, o))
        return state.update(out_shapes=deduplicate_all_shapes(result))

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.x_model, MemorizedModel)
        assert isinstance(self.y_model, MemorizedModel)
        assert isinstance(self.w_model, MemorizedModel)
        assert isinstance(self.c_model, MemorizedModel)
        assert isinstance(self.o_model, MemorizedModel)

        df = default_make_df(state, task.atn)
        x_models = regressor_factory(df, self.x_model.result, self.params, 'rect.x')
        y_models = regressor_factory(df, self.y_model.result, self.params, 'rect.y')
        w_models = regressor_factory(df, self.w_model.result, self.params, 'rect.w')
        c_models = regressor_factory(df, self.c_model.result, self.params, 'rect.c')
        o_models = classifier_factory(df, self.o_model.result, self.params, 'rect.o')
        return [CreateDiagonal(
            x_model, y_model, w_model, c_model, o_model, self.params)
            for x_model, y_model, w_model, c_model, o_model in model_selection(
            x_models, y_models, w_models, c_models, o_models)]
