from ...graphic import *
from ..low_level import *
from typing import Optional
from ...ml import *
import pandas as pd


def find_common_y_shapes(all_y_shapes: list[list[Shape]])->list[Shape]:
    '''Find consistent shapes across all samples.'''
    props = gen_prop(all_y_shapes, False)
    common_shapes = props.groupby('prop').aggregate('first').drop_duplicates()
    common_shapes.sort_values(['sample_id', 'index'], inplace=True)  # type:ignore
    return [all_y_shapes[id1][id2]
            for id1, id2 in zip(common_shapes['sample_id'], common_shapes['index'])]


def make_syntactic_models(sample_index: list[int], x_index: list[list[int]],
                          all_x_shapes: list[list[Shape]])->list[MLModel]:
    '''
    Find shapes outside attention such that:
        1. has a consistent property
        2. appear exactly once per sample
        3. this consistent property cannot be found in any attended shape

    Return the model that find such shapes.
    '''
    unused_df = _make_syntactic_df(all_x_shapes, sample_index, x_index, False)
    if unused_df is None:
        return []

    n_samples = len(all_x_shapes)
    models = _generate_models(unused_df, n_samples)
    used_df = _make_syntactic_df(all_x_shapes, sample_index, x_index, True)
    if used_df is None:
        return models

    return _filter_intersect(models, used_df)


def predict_syntactic_shapes(
        model: MLModel, sample_index: list[int], x_index: list[list[int]],
        all_x_shapes: list[list[Shape]])->Optional[list[int]]:
    '''
    Predict shapes such that:
        1. match the consistent property provided by the model
        2. appear exactly once per sample
        3. does not appear in attention

    If not all properties are met, return None.
    Otherwise, return the indexes of such shapes with the same shape as sample_index.
    '''

    used_df = _make_syntactic_df(all_x_shapes, sample_index, x_index, True)
    if used_df is not None:
        syntactics_in_attention = model.predict_bool(used_df)
        if np.any(syntactics_in_attention):
            return None

    unused_df = _make_syntactic_df(all_x_shapes, sample_index, x_index, False)
    if unused_df is None:
        return None

    syntactics_outside_attention = model.predict_bool(unused_df)
    unused_df['label'] = syntactics_outside_attention
    selected_shapes = unused_df[unused_df['label'] == 1].copy()
    assert isinstance(selected_shapes, pd.DataFrame)
    selected_shapes.sort_values('sample_index', inplace=True)
    if not np.array_equal(selected_shapes['sample_index'], range(len(all_x_shapes))):
        return None

    shape_indexes = list(selected_shapes['shape_index'])
    return [shape_indexes[sample_id] for sample_id in sample_index]


def _make_syntactic_df(
        all_x_shapes: list[list[Shape]], sample_index: list[int],
        x_index: list[list[int]], intersect: bool = False)->Optional[pd.DataFrame]:
    mentioned_index = set()
    for sample_id, indexes in zip(sample_index, x_index):
        for index in indexes:
            mentioned_index.add((sample_id, index))

    relevant_shapes = []
    relevant_sample_index, relevant_shape_index = [], []
    for i, shapes in enumerate(all_x_shapes):
        for j, shape in enumerate(shapes):
            if intersect == ((i, j) in mentioned_index):
                relevant_shapes.append([shape])
                relevant_sample_index.append(i)
                relevant_shape_index.append(j)
    if len(relevant_shapes) == 0:
        return None
    df = generate_df(all_shapes=relevant_shapes)
    df['sample_index']=relevant_sample_index
    df['shape_index']=relevant_shape_index
    return df


def _generate_models(df: pd.DataFrame, n_samples: int)->list[MLModel]:
    models = []
    for col in df.columns:
        if col in ('sample_index', 'shape_index'):
            continue

        filtered_df = df[['sample_index', col]]
        assert isinstance(filtered_df, pd.DataFrame)
        arity_df = filter_constant_arity(filtered_df, ['sample_index'], col, True)
        if arity_df.empty:
            continue
        if len(arity_df) != n_samples:
            continue

        value = arity_df.loc[0, col]
        models.append(ConstantColumnModel(col, value))
    return models


def _filter_intersect(models: list[MLModel], df: pd.DataFrame)->list[MLModel]:
    models2 = []
    for model in models:
        is_syntactics = model.predict_bool(df)
        if not np.any(is_syntactics):
            models2.append(model)
    return models2
