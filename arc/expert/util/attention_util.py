from ...base import *
from ...attention import *
from ...graphic import *
from ...manager.task import *
from functools import lru_cache
import pandas as pd
import numpy as np
from ...ml import *
from typing import Union
from ...global_attention import *

TaskWithAtn = Union[DrawLineTask, TrainingDrawLineTask,
                    AttentionTask, TrainingAttentionTask]


@lru_cache
def default_make_df(
        state: ArcState, task: TaskWithAtn, edit_index: int = -1)->pd.DataFrame:
    '''Make dataframe from relevant information.'''

    assert state.out_shapes != None
    assert state.x_shapes != None
    x = [state.x[i] for i in task.atn.sample_index]
    x_shapes = []

    for sample, shape_indexes in zip(task.atn.sample_index, task.atn.x_index):
        row = [state.out_shapes[sample][i] for i in shape_indexes]
        if isinstance(task, DrawLineTask) or isinstance(task, TrainingDrawLineTask):
            x_shapes.append(row)
            continue

        row.extend(task.common_y_shapes)
        x_shapes.append(row)

    extra_columns: list[ColumnMaker] = []
    if isinstance(task, AttentionTask) or isinstance(task, TrainingAttentionTask):
        extra_columns = [QueryShapeColumn(query, i, state.x_shapes)
                         for i, query in enumerate(task.g_atn.shape_queries)]
    return generate_df(x, x_shapes, edit_index=edit_index, extra_columns=extra_columns)


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


class QueryShapeColumn(ColumnMaker):
    def __init__(self, query: ShapeQuery, index: int,
                 all_shapes: list[list[Shape]])->None:
        self.query = query
        self.index = index
        self.all_shapes = all_shapes

    def append_all(
            self, result: dict[str, list[float]], grids: Optional[list[Grid]],
            all_shapes: Optional[list[list[Shape]]], edit_index: int)->None:
        if self.query.is_null():
            return

        for index, (id1, id2) in enumerate(zip(
                self.query.sample_index, self.query.shape_index)):
            shape = self.all_shapes[id1][id2]
            shape_properties = shape.to_input_var()
            for k, v in shape_properties.items():
                result_key = f'+query{self.index}.{k}'
                result_value = result.get(result_key, None)
                if result_value is None:
                    result_value = result[result_key] = [MISSING_VALUE]*index
                result_value.append(v)
