from ..base import (Task, ModeledTask, InferenceTask, ArcTrainingState,
                    ArcInferenceState, default_cost)
from dataclasses import dataclass
from ..attention import (TrainingAttention, InferenceAttention, Attention,
                         to_models, to_runtimes, AttentionModel)
import pandas as pd
import numpy as np
from typing import Optional
from ..ml import generate_df
from ..graphic import Grid, Shape
from ..constant import GlobalParams


@dataclass(frozen=True)
class AttentionTask(InferenceTask):
    atn: Attention
    common_y_shapes: tuple[Shape, ...]

    def get_cost(self)->int:
        return default_cost(self)


@dataclass(frozen=True)
class TrainingAttentionTask(Task[ArcTrainingState]):
    atn: TrainingAttention
    common_y_shapes: tuple[Shape, ...]
    params: GlobalParams

    def to_models(self, before: ArcTrainingState,
                  after: ArcTrainingState)->list[ModeledTask]:
        assert before.out_shapes is not None
        assert before.x_shapes is not None
        models = to_models(self.atn, before.out_shapes, before.x,
                           before.x_shapes, self.params)
        return [ModeledAttentionTask(model, self.common_y_shapes) for model in models]

    def to_inference(self)->InferenceTask:
        return AttentionTask(self.atn, self.common_y_shapes)


@dataclass(frozen=True)
class ModeledAttentionTask(ModeledTask[ArcInferenceState]):
    model: AttentionModel
    common_y_shapes: tuple[Shape, ...]

    def to_runtimes(self, before: ArcInferenceState)->Optional[InferenceTask]:
        assert before.out_shapes is not None
        assert before.x_shapes is not None
        atn = to_runtimes(self.model, before.out_shapes, before.x, before.x_shapes)
        if atn is None:
            return None
        return AttentionTask(atn, self.common_y_shapes)
