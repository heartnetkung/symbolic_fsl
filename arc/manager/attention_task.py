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

    def get_cost(self)->int:
        return default_cost(self)


@dataclass(frozen=True)
class TrainingAttentionTask(Task[ArcTrainingState]):
    atn: TrainingAttention
    params: GlobalParams

    def to_models(self, before: ArcTrainingState,
                  after: ArcTrainingState)->list[ModeledTask]:
        assert before.out_shapes is not None
        models = to_models(self.atn, before.out_shapes, before.x, self.params)
        return [ModeledAttentionTask(model) for model in models]

    def to_inference(self)->InferenceTask:
        return AttentionTask(self.atn)


@dataclass(frozen=True)
class ModeledAttentionTask(ModeledTask[ArcInferenceState]):
    model: AttentionModel

    def to_runtimes(self, before: ArcInferenceState)->Optional[InferenceTask]:
        assert before.out_shapes is not None
        atn = to_runtimes(self.model, before.out_shapes, before.x)
        if atn is None:
            return None
        return AttentionTask(atn)
