from __future__ import annotations
from ..graphic import *
from ..ml import *
from dataclasses import dataclass, replace, field
from typing import Optional, Union
import pandas as pd
from ..constant import default_hash


@dataclass(frozen=True)
class TrainingAttention:
    '''
    Object encapsulating the result of attention.
    Conceptually, it's a dataframe even though the actual data structure is list.
    Each row represents a causal relationship between multiple x_shapes and a single
    y_shape. That is, these X shapes are relevant in the creation of y.

    Each shape is inderectly represented by sample_index and shape_index.
    To access the actual object, use state.out_train_shapes[sample_index][shape_index].
    '''

    # grids associated with each y_train, having the dimension of (n,)
    sample_index: list[int]
    # index of x_shapes in a sample, having the dimension of (m,n)
    x_index: list[list[int]]
    # index of y_shapes in a sample, having the dimension of (n,)
    y_index: list[int]
    # percentage of relationship between x and y in each cluster
    # example columns: x_label, overlap, same_shape, ...
    # example value: 0, 1, 0.5,...
    relationship_info: pd.DataFrame = field(compare=False)
    # count of columns for each x cluster from left to right
    x_cluster_info: list[int]
    # extra shapes to attend to regardless of row
    extra_shapes: list[Shape]
    # model for predicting extra shapes
    syntactic_model: Optional[MLModel] = None

    def __post_init__(self):
        # check n
        n_rows = len(self.x_index)
        assert len(self.y_index) == len(self.sample_index) == n_rows
        assert n_rows > 0

        # check m
        n_cols = len(self.x_index[0])
        for shape_index in self.x_index:
            assert len(shape_index) == n_cols

        # check syntactic_model
        if self.syntactic_model is None:
            assert sum(self.x_cluster_info) == n_cols
        else:
            assert sum(self.x_cluster_info) + 1 == n_cols

    def update(self, **kwargs)->Attention:
        return replace(self, **kwargs)

    def __hash__(self)->int:
        return default_hash(self)


@dataclass(frozen=True)
class InferenceAttention:
    '''Similar to TrainingAttention but less info'''

    sample_index: list[int]
    x_index: list[list[int]]
    extra_shapes: list[Shape]
    model: MLModel
    syntactic_model: Optional[MLModel]

    def __post_init__(self):
        # check n
        n_rows = len(self.x_index)
        assert len(self.sample_index) == n_rows
        assert n_rows > 0

        # check m
        n_cols = len(self.x_index[0])
        for shape_index in self.x_index:
            assert len(shape_index) == n_cols

    def update(self, **kwargs)->Attention:
        return replace(self, **kwargs)

    def __hash__(self)->int:
        return default_hash(self)


Attention = Union[TrainingAttention, InferenceAttention]


@dataclass(frozen=True)
class AttentionModel:
    '''Extra information required to infer new attentions.'''
    model: MLModel
    x_cluster_info: list[int]
    extra_shapes: list[Shape]
    syntactic_model: Optional[MLModel] = None


def create_empty_attention(
        all_y_shapes: list[list[Shape]])->Optional[TrainingAttention]:
    '''
    Singleton attention is where all y_grids contain exactly one shape.
    In such case, it's possible that there is no attention at all.
    '''
    for y_shapes in all_y_shapes:
        if len(y_shapes) != 1:
            return None

    train_count = len(all_y_shapes)
    x_index = [[]]*train_count
    y_index = [0]*train_count
    sample_index = list(range(train_count))
    empty_df = pd.DataFrame({})
    return TrainingAttention(sample_index, x_index, y_index, empty_df, [], [])
