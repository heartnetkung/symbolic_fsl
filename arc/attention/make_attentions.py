from ..constant import *
from ..graphic import *
from .low_level import *
from .types import *
import pandas as pd
from typing import Optional
from .make_attention.cluster_y import *
from .make_attention.cluster_x import *
from .make_attention.find_shapes import *
from .to_runtime.align_x import *

MAX_ATTENDING_SHAPE = 3


def make_attentions(
        output_train_shapes: list[list[Shape]], y_train_shapes: list[list[Shape]],
        x_train: list[Grid])->list[TrainingAttention]:
    '''
    Create attention from the current training state.
    The returning outer array contains all possible ways to attend to shapes in this state.
    '''

    rel_df = gen_rel_product(output_train_shapes, y_train_shapes)
    possible_y_clusters = cluster_y(rel_df)
    return _make_attentions(output_train_shapes, y_train_shapes,
                            x_train, rel_df, possible_y_clusters)


def is_attention_solved(
        atn: TrainingAttention, output_train_shapes: list[list[Shape]],
        y_train_shapes: list[list[Shape]])->FuzzyBool:
    '''
    Check if the previous attention is resolved.
    If it is solved, the subsequent attention should focus on the next things.
    If it is not, the subsequent attention should still focus on the same thing.
    '''

    relevant_y_shapes = {y_train_shapes[id1][id2] for id1, id2 in zip(
        atn.sample_index, atn.y_index)}
    all_x_shapes = {shape for shapes in output_train_shapes for shape in shapes}
    if relevant_y_shapes.issubset(all_x_shapes):
        return FuzzyBool.yes
    elif len(relevant_y_shapes & all_x_shapes) == 0:
        return FuzzyBool.no
    return FuzzyBool.maybe


def remake_attentions(attention: TrainingAttention,
                      output_train_shapes: list[list[Shape]],
                      y_train_shapes: list[list[Shape]],
                      x_train: list[Grid])->list[TrainingAttention]:
    '''
    Remake the attention with the same y part but with updated value from x parts.
    '''

    rel_df = gen_rel_product(output_train_shapes, y_train_shapes)
    possible_y_clusters = [_copy_cluster_y(attention, rel_df)]
    return _make_attentions(output_train_shapes, y_train_shapes,
                            x_train, rel_df, possible_y_clusters)


# ==============================
# private functions
# ==============================


def _make_attentions(
        output_train_shapes: list[list[Shape]], y_train_shapes: list[list[Shape]],
        x_train: list[Grid], rel_df: pd.DataFrame,
        y_clusters: list[pd.DataFrame])->list[TrainingAttention]:
    result = []
    common_y_shapes = find_common_y_shapes(y_train_shapes)

    for y_cluster in y_clusters:
        possible_x_clusters = cluster_x(rel_df, y_cluster)
        for x_cluster in possible_x_clusters:
            y_cluster_labels = set(x_cluster['y_label'])
            for y_cluster_label in y_cluster_labels:
                current_df = x_cluster[x_cluster['y_label'] == y_cluster_label].copy()
                assert isinstance(current_df, pd.DataFrame)
                if current_df.empty:
                    continue
                result += _make_inner_attentions(
                    current_df, common_y_shapes, output_train_shapes)

    if len(result) == 0:
        empty_attention = create_empty_attention(y_train_shapes)
        if empty_attention is not None:
            return [empty_attention]
    return list(dict.fromkeys(result))


def _make_inner_attentions(df: pd.DataFrame, common_y_shapes: list[Shape],
                           all_shapes: list[list[Shape]])->list[TrainingAttention]:
    grouped_series = df.sort_values(['sample_id', 'y_index', 'x_label']).groupby(
        ['sample_id', 'y_index'])[['x_index']].apply(lambda x: list(x['x_index']))
    index = grouped_series.index.to_frame()
    rel_info = df.iloc[:, 4:].groupby('x_label').mean().reset_index()
    cluster_counts = df.sort_values('x_label').groupby('x_label').size().to_list()

    sample_index = index['sample_id'].to_list()
    y_index = index['y_index'].to_list()
    x_index = grouped_series.to_list()
    x_cluster_info = [int(round(count/len(y_index))) for count in cluster_counts]
    if sum(x_cluster_info) > MAX_ATTENDING_SHAPE:
        return []

    x_index2 = _align_x_index(all_shapes, x_index, sample_index, x_cluster_info)
    syntactic_info = _apply_syntactic(sample_index, x_index2, all_shapes)
    return [TrainingAttention(
        sample_index, x_index3, y_index, rel_info, x_cluster_info, common_y_shapes,
        model) for model, x_index3 in syntactic_info]


def _align_x_index(all_shapes: list[list[Shape]], x_index: list[list[int]],
                   sample_index: list[int], arity: list[int])->list[list[int]]:
    result = []
    for sample_id, index in zip(sample_index, x_index):
        shapes = all_shapes[sample_id]
        result.append(do_align(arity, shapes, index))
    return result


def _apply_syntactic(
        sample_index: list[int], x_index: list[list[int]],
        all_shapes: list[list[Shape]])->list[tuple[Optional[MLModel], list[list[int]]]]:
    models = make_syntactic_models(sample_index, x_index, all_shapes)
    if len(models) == 0:
        return [(None, x_index)]

    result = []
    for model in models:
        prediction = predict_syntactic_shapes(model, sample_index, x_index, all_shapes)
        if prediction is None:
            continue

        x_index2 = [index+[extra] for index, extra in zip(x_index, prediction)]
        result.append((model, x_index2))
    return result


def _copy_cluster_y(atn: TrainingAttention, rel_df: pd.DataFrame)->pd.DataFrame:
    filter_data = {'sample_id': atn.sample_index, 'y_index': atn.y_index}
    filter_df = pd.DataFrame(filter_data)
    result = to_y_df(rel_df.merge(filter_df, on=['sample_id', 'y_index'], how='inner'))
    result['y_label'] = 0
    return result
