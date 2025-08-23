from .make_rel_df import make_rel_df
from ...graphic import *
import pandas as pd
import numpy as np
from ...ml import *
from ..types import *
from ...constant import MAX_SHAPES_PER_GRID


MAX_RETURN = 2


def make_shape_queries(
        grids: list[Grid], all_shapes: list[list[Shape]])->list[ShapeQuery]:
    assert len(grids) == len(all_shapes)
    for shapes in all_shapes:
        if len(shapes) > MAX_SHAPES_PER_GRID:
            return []

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
        sample_index = tuple(range(_len))
        shape_index = tuple(index)
        models = tuple(ColumnModel(rel) for rel in rels)
        results.append(ShapeQuery(sample_index, shape_index, models))
    results.sort(key=lambda query: -len(query.models))
    return results[:MAX_RETURN]


def make_query_df(grids: list[Grid], all_shapes: list[list[Shape]])->pd.DataFrame:
    df = make_rel_df(grids, all_shapes)
    result = df.pivot_table(index=['sample_index', 'shape_index'],
                            columns='rel', aggfunc='size').fillna(0).reset_index()
    return result


def check_query_result(df: pd.DataFrame, count: int, selection: list[bool])->bool:
    selected_rows = df[selection]
    return ((len(selected_rows) == count) and
            (len(set(selected_rows['sample_index'])) == count))
