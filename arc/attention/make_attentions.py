from ..constant import *
from ..graphic import *
from .low_level import *
from .attention_obj import *
import pandas as pd
from typing import Optional
from .make_attention.cluster_y import *
from .make_attention.cluster_x import *
from .make_attention.find_shapes import *


def make_attentions(
        output_train_shapes: list[list[Shape]], y_train_shapes: list[list[Shape]],
        X_train: list[Grid])->list[Attention]:
    '''
    Create attention from the current training state.
    The returning outer array contains all possible ways to attend to shapes in this state.
    '''

    rel_df = gen_rel_product(output_train_shapes, y_train_shapes)
    possible_y_clusters = cluster_y(rel_df)
    return _make_attentions(output_train_shapes, y_train_shapes,
                            X_train, rel_df, possible_y_clusters)


def is_attention_solved(
        atn: Attention, output_train_shapes: list[list[Shape]],
        y_train_shapes: list[list[Shape]])->FuzzyBool:
    '''
    Check if the previous attention is resolved.
    If it is solved, the subsequent attention should focus on the next things.
    If it is not, the subsequent attention should still focus on the same thing.
    '''

    relevant_y_shapes = {y_train_shapes[id1][id2] for id1, id2 in zip(
        atn.sample_train_index, atn.y_train_index)}
    all_x_shapes = {shape for shapes in output_train_shapes for shape in shapes}
    if relevant_y_shapes.issubset(all_x_shapes):
        return FuzzyBool.yes
    elif len(relevant_y_shapes & all_x_shapes) == 0:
        return FuzzyBool.no
    return FuzzyBool.maybe


def remake_attentions(
        attention: Attention, output_train_shapes: list[list[Shape]],
        y_train_shapes: list[list[Shape]], X_train: list[Grid])->list[Attention]:
    '''
    Remake the attention with the same y part but with updated value from x parts.
    '''

    rel_df = gen_rel_product(output_train_shapes, y_train_shapes)
    possible_y_clusters = [_copy_cluster_y(attention, rel_df)]
    return _make_attentions(output_train_shapes, y_train_shapes,
                            X_train, rel_df, possible_y_clusters)


# ==============================
# private functions
# ==============================


def _make_attentions(
        output_train_shapes: list[list[Shape]], y_train_shapes: list[list[Shape]],
        X_train: list[Grid], rel_df: pd.DataFrame,
        y_clusters: list[pd.DataFrame])->list[Attention]:
    result = []

    for y_cluster in y_clusters:
        possible_x_clusters = cluster_x(rel_df, y_cluster)
        for x_cluster in possible_x_clusters:
            y_cluster_labels = set(x_cluster['y_label'])
            for y_cluster_label in y_cluster_labels:
                current_df = x_cluster[x_cluster['y_label'] == y_cluster_label].copy()
                assert isinstance(current_df, pd.DataFrame)
                if current_df.empty:
                    continue
                result.append(_make_attention(current_df))

    if len(result) == 0:
        singleton = create_singleton_attention(y_train_shapes)
        if singleton is not None:
            result.append(singleton)
    return list(dict.fromkeys(result))


def _make_attention(df: pd.DataFrame)->Optional[Attention]:
    grouped_series = df.sort_values(['sample_id', 'y_index', 'x_label']).groupby(
        ['sample_id', 'y_index'])[['x_index']].apply(lambda x: list(x['x_index']))
    index = grouped_series.index.to_frame()
    rel_info = df.iloc[:, 4:].groupby('x_label').mean().reset_index()

    sample_train_index = index['sample_id'].to_list()
    y_train_index = index['y_index'].to_list()
    X_train_index = grouped_series.to_list()
    return Attention(sample_train_index, X_train_index, y_train_index, rel_info)


def _copy_cluster_y(atn: Attention, rel_df: pd.DataFrame)->pd.DataFrame:
    filter_data = {'sample_id': atn.sample_train_index, 'y_index': atn.y_train_index}
    filter_df = pd.DataFrame(filter_data)
    result = to_y_df(rel_df.merge(filter_df, on=['sample_id', 'y_index'], how='inner'))
    result['y_label'] = 0
    return result
