from ...base import *
from ...attention import *
from ...graphic import *
from ...manager.task import *
from functools import lru_cache
import pandas as pd
import numpy as np
from ...ml import *


@lru_cache
def default_make_df(
        state: ArcState, atn: Attention, edit_index: int = -1)->pd.DataFrame:
    '''Make dataframe from relevant information.'''

    assert state.out_shapes != None
    x = [state.x[i] for i in atn.sample_index]
    x_shapes = []
    for sample, shape_indexes in zip(atn.sample_index, atn.x_index):
        row = [state.out_shapes[sample][i] for i in shape_indexes]
        x_shapes.append(row)
    return generate_df(x, x_shapes, edit_index=edit_index)


def has_relationship(atn: TrainingAttention, relationship: str, column: int)->bool:
    '''Check if all x shapes in a column has a relationship of a given type.'''
    try:
        x_label = 0
        for cluster_column in atn.x_cluster_info:
            if column < cluster_column:
                break
            column -= cluster_column
            x_label += 1
        info = atn.relationship_info
        filtered = info.loc[info['x_label'] == x_label, relationship]
        return (len(filtered) > 0) and np.allclose(filtered, 1)
    except KeyError:
        return False


def list_editable_feature_indexes(atn: TrainingAttention)->list[int]:
    '''List feature columns that are unique.'''
    result = []
    n_rows, n_cols = len(atn.x_index), len(atn.x_index[0])
    for feat_index in range(n_cols):
        unique_indexes = {(sample_id, atn.x_index[i][feat_index])
                          for i, sample_id in enumerate(atn.sample_index)}
        if len(unique_indexes) == n_rows:
            result.append(feat_index)
    return result


def get_y_shapes(state: ArcTrainingState, atn: TrainingAttention)->list[Shape]:
    assert state.y_shapes is not None
    return [state.y_shapes[id1][id2]
            for id1, id2 in zip(atn.sample_index, atn.y_index)]


def get_x_col(state: ArcState, atn: Attention, feat_index: int)->list[Shape]:
    assert state.out_shapes is not None
    return [state.out_shapes[id1][index[feat_index]]
            for id1, index in zip(atn.sample_index, atn.x_index)]


def get_grids(state: ArcState, atn: Attention)->list[Grid]:
    return [state.x[id1] for id1 in atn.sample_index]
