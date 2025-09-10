from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from copy import deepcopy
from ..util import *
from enum import Enum
from .draw_intersect_processor import *


class DrawIntersect(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, feat_index: int, old_color_model: MLModel,
                 new_color_model: MLModel, unchanged_color: int,
                 params: GlobalParams)->None:
        self.feat_index = feat_index
        self.old_color_model = old_color_model
        self.new_color_model = new_color_model
        self.unchanged_color = unchanged_color
        self.params = params

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes is not None

        atn = task.atn
        result = deepcopy(state.out_shapes)
        shape_df = default_make_df(state, task, self.feat_index)
        old_colors = self.old_color_model.predict_int(shape_df)
        new_colors = self.new_color_model.predict_int(shape_df)

        for id1, shape_ids, old_color, new_color in zip(
                atn.sample_index, atn.x_index, old_colors, new_colors):
            id2 = shape_ids[self.feat_index]
            shape = result[id1][id2]
            new_shape = self._perform(shape, old_color, new_color)
            if new_shape is None:
                return None

            result[id1][id2] = new_shape
        return state.update(out_shapes=result)

    def _perform(self, shape: Shape, old_color: int, new_color: int)->Optional[Shape]:
        processor = DrawIntersectProcessor(old_color, new_color, self.unchanged_color)
        candidates = processor.process(shape._grid)
        if candidates is None:
            return None

        result_grid = Grid(deepcopy(shape._grid.data))
        for candidate in candidates:
            candidate.draw(result_grid)
        return Unknown(shape.x, shape.y, result_grid)

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.old_color_model, MemorizedModel)
        assert isinstance(self.new_color_model, MemorizedModel)

        df = default_make_df(state, task, self.feat_index)
        labels = [self.old_color_model.result, self.new_color_model.result]
        label_types = [LabelType.reg]*2
        all_models = make_all_models(
            df, self.params, 'draw_intersect', labels, label_types)
        return [DrawIntersect(self.feat_index, oldc_model, newc_model,
                              self.unchanged_color, self.params)
                for oldc_model, newc_model in model_selection(*all_models)]
