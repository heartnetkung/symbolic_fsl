from ..graphic import *
from ..base import *
from dataclasses import dataclass
from typing import Union
from ..ml import ColumnModel


@dataclass(frozen=True)
class TrainingGlobalAttention:
    # all these pointers is used with x_shapes not out_shapes
    query_sample_index: list[int]
    query_shape_index: list[int]
    query_models: list[ColumnModel]


@dataclass(frozen=True)
class GlobalAttentionModel:
    query_model: ColumnModel


@dataclass(frozen=True)
class InferenceGlobalAttention:
    # all these pointers is used with x_shapes not out_shapes
    query_sample_index: list[int]
    query_shape_index: list[int]
    query_model: ColumnModel


GlobalAttention = Union[TrainingGlobalAttention, InferenceGlobalAttention]
