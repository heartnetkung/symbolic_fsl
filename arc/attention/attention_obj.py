from __future__ import annotations
from ..graphic import *
from ..ml import *
from dataclasses import dataclass, replace, field, fields, asdict
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class Attention:
    '''
    Object encapsulating the result of attention.
    Conceptually, it's a dataframe even though the actual data structure is list.
    Each row represents a causal relationship between multiple X_train shapes and a single
    y_train shape. That is, these X shapes are relevant in the creation of y.

    Each shape is inderectly represented by sample_index and shape_index.
    To access the actual object, use state.out_train_shapes[sample_index][shape_index].
    '''

    # grids associated with each y_train, having the dimension of (n,)
    sample_train_index: list[int]
    # index of X_train shapes in a sample, having the dimension of (m,n)
    X_train_index: list[list[int]]
    # index of y shapes in a sample, having the dimension of (n,)
    y_train_index: list[int]
    # percentage of relationship between x and y in each cluster
    # example columns: x_label, overlap, same_shape, ...
    # example value: 0, 1, 0.5,...
    relationship_info: pd.DataFrame = field(compare=False)

    # model used to predict this attention (may be used for complexity calculation)
    model: Optional[MLModel] = field(compare=False, default=None)
    # common shapes found across y samples
    common_y_shapes: Optional[list[Shape]] = None

    # grids associated with each y_pred, having the dimension of (n2,)
    sample_test_index: Optional[list[int]] = None
    # index of X_test shapes in a sample, having the dimension of (m,n2)
    X_test_index: Optional[list[list[int]]] = None

    def __post_init__(self):
        # check n
        assert len(self.y_train_index) == len(
            self.sample_train_index) == self.n_train_rows
        assert self.n_train_rows > 0
        # check n2
        if self.sample_test_index is not None:
            assert len(self.sample_test_index) == self.n_test_rows
            assert self.n_test_rows > 0
        # check m
        n_cols = self.n_cols
        for shape_index in self.X_train_index:
            assert len(shape_index) == n_cols
        if self.X_test_index is not None:
            for shape_index in self.X_test_index:
                assert len(shape_index) == n_cols

    @property
    def n_train_rows(self)->int:
        return len(self.X_train_index)

    @property
    def n_test_rows(self)->int:
        if self.X_test_index is None:
            return 0
        return len(self.X_test_index)

    @property
    def n_cols(self)->int:
        return len(self.X_train_index[0])

    def update(self, **kwargs)->Attention:
        return replace(self, **kwargs)

    def __hash__(self)->int:
        comparable_fields = [f.name for f in fields(self) if f.compare]
        dict_ = asdict(self)
        values = tuple(repr(dict_[f]) for f in comparable_fields)
        return hash(values)


def create_singleton_attention(all_y_shapes: list[list[Shape]])->Optional[Attention]:
    '''
    Singleton attention is where all y_grids contain exactly one shape.
    In such case, it's possible that there is no attention at all.
    '''
    for y_shapes in all_y_shapes:
        if len(y_shapes) != 1:
            return None

    train_count = len(all_y_shapes)
    X_train_index = [[]]*train_count
    y_train_index = [0]*train_count
    sample_train_index = list(range(train_count))
    empty_df = pd.DataFrame({})
    return Attention(
        sample_train_index, X_train_index, y_train_index, empty_df)
