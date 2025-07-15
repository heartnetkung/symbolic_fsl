from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from ...attention import *
from copy import deepcopy
import pandas as pd
from ..util import *


class Move(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, x_model: MLModel, y_model: MLModel, params: GlobalParams,
                 feat_index: int = 0)->None:
        self.x_model = x_model
        self.y_model = y_model
        self.params = params
        self.feat_index = feat_index
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes != None

        atn = task.atn
        df = default_make_df(state, atn, self.feat_index)
        x_values = self.x_model.predict_int(df)
        y_values = self.y_model.predict_int(df)
        result = deepcopy(state.out_shapes)
        for id1, shape_ids, x, y in zip(
                atn.sample_index, atn.x_index, x_values, y_values):
            id2 = shape_ids[self.feat_index]
            current_shape = deepcopy(result[id1][id2])
            current_shape.x = x
            current_shape.y = y
            result[id1][id2] = current_shape
        return state.update(out_shapes=result)

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        if not isinstance(self.x_model, MemorizedModel):
            return []
        if not isinstance(self.y_model, MemorizedModel):
            return []

        df = default_make_df(state, task.atn, self.feat_index)
        x_models = regressor_factory(df, self.x_model.result, self.params, 'move.x')
        y_models = regressor_factory(df, self.y_model.result, self.params, 'move.y')
        return [Move(x_model, y_model, self.params, self.feat_index)
                for x_model, y_model in model_selection(x_models, y_models)]
