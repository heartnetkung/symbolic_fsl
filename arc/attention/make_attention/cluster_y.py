from ...constant import *
from ...graphic import *
from ..low_level import *
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN


def cluster_y(rel_df: pd.DataFrame)->list[pd.DataFrame]:
    '''
    Cluster y shapes based on their consistent relationship with x.
    For a cluster to be valid, it must be consistent and surjective.
    The result is all possible cluster formation.
    '''
    y_df, feat_arr = _preprocess(rel_df)
    return _cluster_y(y_df, feat_arr)


def _preprocess(rel_df: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
    y_df = to_y_df(rel_df)
    return y_df, y_df.iloc[:, 2:]


def _cluster_y(y_df: pd.DataFrame, feat_arr: pd.DataFrame)->list[pd.DataFrame]:
    min_samples = cal_consistency_requirement(y_df)
    models = [OPTICS(min_samples=min_samples, metric='precomputed', max_eps=99),
              DBSCAN(min_samples=min_samples, metric='precomputed')]
    metrics = ['l1', 'braycurtis']

    results = {}
    for model, metric in product(models, metrics):
        try:
            with np.errstate(divide='ignore'):
                model.fit(to_distance_matrix(feat_arr, metric))
        except ValueError:
            continue

        unclustered = model.labels_ == -1
        if np.any(unclustered):  # surjective relation
            continue

        new_result = y_df.copy()
        new_result['y_label'] = model.labels_
        if len(new_result) < 2:
            continue

        # valid cluster must have consistent labels
        if _is_cluster_consistent(new_result):
            results[repr(model.labels_)] = new_result

    return list(results.values())


def to_y_df(rel_df: pd.DataFrame)->pd.DataFrame:
    dummies = pd.get_dummies(rel_df['rel'], dtype='float')
    concatted = pd.concat([rel_df, dummies], axis=1)
    one_hot_encoded = concatted.groupby(['sample_id', 'y_index']).sum()
    return one_hot_encoded.reset_index().drop(columns=['rel', 'x_index'])


def _is_cluster_consistent(y_df: pd.DataFrame):
    filtered = filter_consistency(y_df, 'y_label')
    return y_df.shape == filtered.shape
