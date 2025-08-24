from ..base import (Task, ModeledTask, InferenceTask, ArcTrainingState,
                    ArcInferenceState, default_cost)
from dataclasses import dataclass
from ..attention import (TrainingAttention, InferenceAttention, Attention,
                         to_models, to_runtimes, AttentionModel)
from ..global_attention import (
    TrainingGlobalAttention, GlobalAttention, GlobalAttentionModel,
    InferenceGlobalAttention, make_gattention, to_gmodel, to_gruntimes)
import pandas as pd
import numpy as np
from typing import Optional
from ..ml import generate_df
from ..graphic import Grid, Shape
from ..constant import GlobalParams


@dataclass(frozen=True)
class AttentionTask(InferenceTask):
    atn: Attention
    g_atn: GlobalAttention
    common_y_shapes: tuple[Shape, ...]

    def get_cost(self)->int:
        return default_cost(self)


@dataclass(frozen=True)
class TrainingAttentionTask(Task[ArcTrainingState]):
    atn: TrainingAttention
    g_atn: TrainingGlobalAttention
    common_y_shapes: tuple[Shape, ...]
    params: GlobalParams

    def to_models(self, before: ArcTrainingState,
                  after: ArcTrainingState)->list[ModeledTask]:
        assert before.out_shapes is not None
        assert before.x_shapes is not None
        models = to_models(self.atn, before.out_shapes, before.x,
                           before.x_shapes, self.params, self.g_atn)
        g_model = to_gmodel(self.g_atn, before.x_shapes, before.x)
        return [ModeledAttentionTask(model, g_model, self.common_y_shapes)
                for model in models]

    def to_inference(self)->InferenceTask:
        return AttentionTask(self.atn, self.g_atn, self.common_y_shapes)


@dataclass(frozen=True)
class ModeledAttentionTask(ModeledTask[ArcInferenceState]):
    model: AttentionModel
    g_model: GlobalAttentionModel
    common_y_shapes: tuple[Shape, ...]

    def to_runtimes(self, before: ArcInferenceState)->list[InferenceTask]:
        assert before.out_shapes is not None
        assert before.x_shapes is not None

        g_atns = to_gruntimes(self.g_model, before.x_shapes, before.x)
        result = []
        for g_atn in g_atns:
            atn = to_runtimes(self.model, before.out_shapes,
                              before.x, before.x_shapes, g_atn)
            if atn is not None:
                result.append(AttentionTask(atn, g_atn, self.common_y_shapes))
        return result
