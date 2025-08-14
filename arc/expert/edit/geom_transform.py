from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from ..util import *
from copy import deepcopy
from enum import Enum


class TransformType(Enum):
    normal = 0
    flip_h = 1
    flip_v = 2
    flip_both = 3
    normal_transpose = 4
    flip_h_transpose = 5
    flip_v_transpose = 6
    flip_both_transpose = 7

    def is_transpose(self)->bool:
        return self.value > 3


class GeomTransform(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, model: MLModel, params: GlobalParams, feat_index: int = 0)->None:
        self.model = model
        self.params = params
        self.feat_index = feat_index
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes != None

        atn = task.atn
        df = default_make_df(state, task, self.feat_index)
        transforms = self.model.predict_enum(df, TransformType)
        result = deepcopy(state.out_shapes)
        for id1, shape_ids, transform in zip(atn.sample_index, atn.x_index, transforms):
            id2 = shape_ids[self.feat_index]
            result[id1][id2] = transform_shape(result[id1][id2], transform)
        return state.update(out_shapes=result)

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.model, MemorizedModel)

        df = default_make_df(state, task, self.feat_index)
        models = make_classifier(df, self.model.result, self.params, 'geom')
        return [GeomTransform(model, self.params, self.feat_index) for model in models]


def transform_shape(shape: Shape, mode: TransformType)->Shape:
    grid = shape._grid
    if mode in (TransformType.flip_h, TransformType.flip_h_transpose):
        grid = grid.flip_h()
    elif mode in (TransformType.flip_v, TransformType.flip_v_transpose):
        grid = grid.flip_v()
    elif mode in (TransformType.flip_both, TransformType.flip_both_transpose):
        grid = grid.flip_both()
    if mode.is_transpose():
        grid = grid.transpose()
    return Unknown(shape.x, shape.y, grid)
