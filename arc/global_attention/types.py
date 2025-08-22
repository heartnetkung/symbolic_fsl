from ..graphic import *
from ..base import *
from dataclasses import dataclass
from typing import Union
from ..ml import ColumnModel


@dataclass(frozen=True)
class ShapeQuery:
    '''A noticable shape with consistent relationship, one for each sample.'''
    sample_index: tuple[int, ...]
    shape_index: tuple[int, ...]
    models: tuple[ColumnModel, ...]


@dataclass(frozen=True)
class TrainingGlobalAttention:
    shape_queries: tuple[ShapeQuery, ...]


@dataclass(frozen=True)
class GlobalAttentionModel:
    query_models: tuple[tuple[ColumnModel, ...], ...]


@dataclass(frozen=True)
class InferenceGlobalAttention:
    shape_queries: tuple[ShapeQuery, ...]


def create_null_shape_query(len_: int)->ShapeQuery:
    null_index = tuple([-1]*len_)
    return ShapeQuery(null_index, null_index, tuple())


GlobalAttention = Union[TrainingGlobalAttention, InferenceGlobalAttention]
