from .make_rel_df import make_rel_df
from ...graphic import *
import pandas as pd
import numpy as np
from dataclasses import dataclass
from ...ml import *


@dataclass(frozen=True)
class QueryResult:
    sample_index: list[int]
    shape_index: list[int]
    models: list[ColumnModel]


def make_shape_queries(
        grids: list[Grid], all_shapes: list[list[Shape]])->list[QueryResult]:
    assert len(grids) == len(all_shapes)
    df = make_rel_df(grids, all_shapes)
    df.sort_values('sample_index', inplace=True)
    groups = df.groupby('rel').agg({'sample_index': ['size', 'nunique']}).reset_index()
    groups.columns = ['rel', 'count', 'n_unique']

    cluster, _len = {}, len(grids)
    for i, row in groups.iterrows():
        if (row['count'] != _len) or (row['n_unique'] != _len):
            continue

        shape_index = tuple(df[df['rel'] == row['rel']]['shape_index'])
        values = cluster.get(shape_index, None)
        if values is None:
            cluster[shape_index] = [row['rel']]
        else:
            values.append(row['rel'])

    results = []
    for index, rels in cluster.items():
        sample_index = list(range(_len))
        shape_index = list(index)
        models = [ColumnModel(rel) for rel in rels]
        results.append(QueryResult(sample_index, shape_index, models))
    return results


def make_query_df(grids: list[Grid], all_shapes: list[list[Shape]])->pd.DataFrame:
    df = make_rel_df(grids, all_shapes)
    result = df.pivot_table(index=['sample_index', 'shape_index'],
                            columns='rel', aggfunc='size').fillna(0).reset_index()
    return result


def check_query_result(df: pd.DataFrame, count: int, selection: list[bool])->bool:
    selected_rows = df[selection]
    return ((len(selected_rows) == count) and
            (len(set(selected_rows['sample_index'])) == count))
