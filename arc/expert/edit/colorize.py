from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from copy import deepcopy
from ..util import *


class Colorize(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, color_model: MLModel, params: GlobalParams,
                 feat_index: int = 0)->None:
        self.color_model = color_model
        self.params = params
        self.feat_index = feat_index
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes != None

        atn = task.atn
        df = default_make_df(state, task, self.feat_index)
        colors = self.color_model.predict_int(df)
        result = deepcopy(state.out_shapes)
        for id1, shape_ids, color in zip(atn.sample_index, atn.x_index, colors):
            id2 = shape_ids[self.feat_index]
            result[id1][id2] = colorize(result[id1][id2], color)
        return state.update(out_shapes=result)

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.color_model, MemorizedModel)

        df = default_make_df(state, task, self.feat_index)
        models = regressor_factory(df, self.color_model.result, self.params, 'colorize')
        return [Colorize(model, self.params, self.feat_index) for model in models]
