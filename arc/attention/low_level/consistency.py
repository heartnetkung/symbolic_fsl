import numpy as np
import pandas as pd
from ...graphic import *
from itertools import product, combinations
from .prop import *
from .rel import *
from sklearn.metrics.pairwise import pairwise_distances
from ...constant import MAX_SHAPES_PER_GRID
from typing import Optional


def gen_rel_product(all_x_shapes: list[list[Shape]],
                    all_y_shapes: list[list[Shape]])->Optional[pd.DataFrame]:
    result = {'sample_id': [], 'x_index': [], 'y_index': [], 'rel': []}
    exclude_exact = _should_exclude_exact(all_x_shapes, all_y_shapes)

    for sample_id, (x_shapes, y_shapes) in enumerate(zip(all_x_shapes, all_y_shapes)):
        if len(x_shapes) > MAX_SHAPES_PER_GRID:
            return None
        if len(y_shapes) > MAX_SHAPES_PER_GRID:
            return None

        x_shapes_set = set(x_shapes)
        for x_index, y_index in product(range(len(x_shapes)), range(len(y_shapes))):
            x_shape, y_shape = x_shapes[x_index], y_shapes[y_index]
            if (y_shape in x_shapes_set) and exclude_exact:
                continue

            for rel in list_relationship(x_shape, y_shape):
                result['sample_id'].append(sample_id)
                result['x_index'].append(x_index)
                result['y_index'].append(y_index)
                result['rel'].append(rel)
    return filter_consistency(pd.DataFrame(result), 'rel')


def _should_exclude_exact(all_x_shapes: list[list[Shape]],
                          all_y_shapes: list[list[Shape]])->int:
    exact_count = 0
    for x_shapes, y_shapes in zip(all_x_shapes, all_y_shapes):
        exact_count += len(set(x_shapes) & set(y_shapes))
    return exact_count >= len(all_x_shapes)


def cal_consistency_requirement(df: pd.DataFrame)->int:
    n_samples = len(pd.unique(df['sample_id']))
    # TODO should we do this? return min(n_samples, 3)
    return n_samples


def filter_consistency(df: pd.DataFrame, col_name: str)->pd.DataFrame:
    '''
    Filter the dataframe for rows that row[col_name] has consistent values.
    Consistency value occurs when such values appear across enough samples.
    '''
    if df.empty:
        return df

    min_appearance = cal_consistency_requirement(df)
    grouped_series = df.groupby(col_name)[['sample_id']].apply(
        lambda x: len(set(x['sample_id'])))
    consistent_filter = grouped_series[
        grouped_series >= min_appearance].reset_index().drop(columns=0)  # type:ignore
    return df.merge(consistent_filter, on=col_name, how='inner')


def filter_constant_arity(df: pd.DataFrame, unit_cols: list[str],
                          value_col: str, exactly_one: bool = False)->pd.DataFrame:
    '''
    Filter the dataframe for rows that row[value_cols]
    has consistent arity per given row[unit_cols].
    Constant arity is that certain data appears exactly n times per unit.
    '''
    if df.empty:
        return df

    pivot = pd.pivot_table(df, index=unit_cols, columns=value_col, aggfunc='size')
    keep_values = []
    for col in pivot.columns:
        if exactly_one and (np.allclose(pivot[col], 1)):
            keep_values.append(col)
        elif (not exactly_one) and (len(pd.unique(pivot[col])) == 1):
            keep_values.append(col)

    result = df[df[value_col].isin(keep_values)].reset_index(drop=True)
    assert isinstance(result, pd.DataFrame)
    return result


def to_distance_matrix(df: pd.DataFrame, metric: str)->np.ndarray:
    # There are many metric types as listed below
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    # [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].

    # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    # [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’,
    # ‘hamming’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’,
    # ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
    # ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    return pairwise_distances(df, metric=metric)
