from ...constant import *
from ...graphic import *
from ..low_level import *
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN
from itertools import product
from typing import Optional

MAX_PERMUTATION = 50


def cluster_x(rel_df: pd.DataFrame, y_cluster: pd.DataFrame)->list[pd.DataFrame]:
    '''
    For each y cluster, cluster x shapes based on their relationship with y.
    Valid x clusters must form constant arity relationship.
    Note that some x shapes might be excluded if it's judged to be noise.
    The result is all possible cluster formation.
    '''

    y_cluster_count = len(pd.unique(y_cluster['y_label']))
    grouped_results = {y_label: {} for y_label in range(y_cluster_count)}
    data_blob = _preprocess(rel_df, y_cluster)

    for y_label, (x_df, feat_arr) in data_blob.items():
        x_labels = _cluster_x(x_df, feat_arr)
        for x_label in x_labels:
            grouped_results[y_label][repr(x_label['x_label'])] = x_label

    return _generate_permutation(grouped_results)


def _cluster_x(x_df: pd.DataFrame, feat_arr: pd.DataFrame)->list[pd.DataFrame]:
    min_samples = cal_consistency_requirement(x_df)
    models = [OPTICS(min_samples=min_samples, metric='precomputed', max_eps=99),
              DBSCAN(min_samples=min_samples, metric='precomputed')]
    metrics = ['l1', 'braycurtis', 'correlation']

    results = {}
    for model, metric in product(models, metrics):
        try:
            with np.errstate(divide='ignore'):
                model.fit(to_distance_matrix(feat_arr, metric))
        except ValueError:
            continue

        new_result = x_df.copy()
        new_result['x_label'] = model.labels_
        new_result = new_result[new_result['x_label'] != -1].reset_index(drop=True)
        assert isinstance(new_result, pd.DataFrame)

        new_result2 = _filter_constant_arity(new_result)
        if new_result2 is not None:
            key = repr(new_result2['x_label'])
            results[key] = _fix_x_label(new_result2)
    return list(results.values())


def _preprocess(rel_df: pd.DataFrame,
                y_cluster: pd.DataFrame)->dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    x_dfs = _to_x_df(rel_df, y_cluster)
    results = {}
    for key, x_df in x_dfs.items():
        exclude_columns = ['sample_id', 'y_index', 'x_index', 'y_label']
        results[key] = (x_df, x_df.drop(columns=exclude_columns))
    return results


def _fix_x_label(df: pd.DataFrame)->pd.DataFrame:
    '''
    x_label tends to skip numbers which is problematic to work with.
    Thus, we relabel to not skip. Example 0,2,4,5 -> 0,1,2,3
    '''
    values = df['x_label'].to_list()
    fix_table = {v: i for i, v in enumerate(dict.fromkeys(sorted(values)))}
    return df.assign(x_label=df['x_label'].replace(fix_table))


def _to_x_df(rel_df: pd.DataFrame, y_cluster: pd.DataFrame)->dict[int, pd.DataFrame]:
    n_labels = len(pd.unique(y_cluster['y_label']))
    results = {}

    for label in range(n_labels):
        loop_mask = y_cluster['y_label'] == label
        filter_df = y_cluster[loop_mask][['sample_id', 'y_index', 'y_label']]
        assert isinstance(filter_df, pd.DataFrame)

        filtered_df = rel_df.merge(filter_df, on=['sample_id', 'y_index'], how='inner')
        dummies = pd.get_dummies(filtered_df['rel'], dtype='float')
        concatted = pd.concat([filtered_df, dummies], axis=1)
        one_hot_encoded = concatted.groupby(
            ['sample_id', 'y_index', 'x_index', 'y_label']).sum()
        new_result = one_hot_encoded.reset_index().drop(columns='rel')
        results[label] = new_result
    return results


def _filter_constant_arity(x_cluster: pd.DataFrame)->Optional[pd.DataFrame]:
    unit_cols, value_col = ['sample_id', 'y_index'], 'x_label'
    filter_columns = ['sample_id', 'y_index', 'x_label']
    cluster_view = x_cluster[filter_columns]
    assert isinstance(cluster_view, pd.DataFrame)
    filtered = filter_constant_arity(cluster_view, unit_cols, value_col)
    if filtered.empty:
        return None

    return x_cluster.merge(filtered, on=filter_columns, how='inner').drop_duplicates()


def _generate_permutation(grouped: dict[int, dict[str, pd.DataFrame]])->list[
        pd.DataFrame]:
    if len(grouped) == 0:
        return []

    all_x_dfs, permutation_count = [], 1
    for group in grouped.values():
        combination = list(group.values())
        all_x_dfs.append(combination)
        permutation_count *= len(combination)

    if permutation_count > MAX_PERMUTATION:
        return []

    results = []
    for df_combination in product(*all_x_dfs):
        results.append(pd.concat(df_combination, axis=0, ignore_index=True).fillna(0))
    return results
