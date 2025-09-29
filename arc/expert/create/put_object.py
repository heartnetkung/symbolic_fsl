from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from ...attention import *
from copy import deepcopy
import pandas as pd
from ..util import *


class PutObject(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, selection_model: MLModel, x_model: MLModel, y_model: MLModel,
                 params: GlobalParams)->None:
        self.selection_model = selection_model
        self.x_model = x_model
        self.y_model = y_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes != None

        df = default_make_df(state, task)
        selections = self.selection_model.predict_int(df)
        x_values = self.x_model.predict_int(df)
        y_values = self.y_model.predict_int(df)
        result = deepcopy(state.out_shapes)

        for id1, selection, x, y in zip(
                task.atn.sample_index, selections, x_values, y_values):
            try:
                prototype = task.common_y_shapes[selection]
                new_shape = deepcopy(prototype)
                new_shape.x = x
                new_shape.y = y
                result[id1].append(new_shape)
            except IndexError:
                return None
        return state.update(out_shapes=deduplicate_all_shapes(result))

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.selection_model, MemorizedModel)
        assert isinstance(self.x_model, MemorizedModel)
        assert isinstance(self.y_model, MemorizedModel)

        df = default_make_df(state, task)
        labels = [self.selection_model.result, self.x_model.result, self.y_model.result]
        label_types = [LabelType.cls_, LabelType.reg, LabelType.reg]
        all_models = make_all_models(df, self.params, 'put_obj', labels, label_types)
        return [PutObject(s_model, x_model, y_model, self.params)
                for s_model, x_model, y_model in model_selection(*all_models)]
