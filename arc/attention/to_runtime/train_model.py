from ...graphic import *
from ..low_level import *
from ...ml import *
import numpy as np
import pandas as pd


def create_label(correct_sample_index: list[int], correct_x_index: list[list[int]],
                 possible_sample_index: list[int],
                 possible_x_index: list[list[int]])->np.ndarray:
    assert len(correct_sample_index) == len(correct_x_index)
    assert len(possible_sample_index) == len(possible_x_index)

    correct_pairs = set()
    for sample_id, x_index in zip(correct_sample_index, correct_x_index):
        correct_pairs.add(repr((sample_id, x_index)))

    return np.array([
        repr((sample_id, x_index)) in correct_pairs
        for sample_id, x_index in zip(possible_sample_index, possible_x_index)])


def create_df(grids: list[Grid], all_shapes: list[list[Shape]],
              sample_index: list[int], all_x_index: list[list[int]],
              original_shapes: list[list[Shape]])->pd.DataFrame:
    extra_features = {f'shape{col}.is_original': []
                      for col in range(len(all_x_index[0]))}
    original_cache = [set(shapes) for shapes in original_shapes]

    resolved_grids, all_resolved_shapes = [], []
    for sample_id, x_index in zip(sample_index, all_x_index):
        resolved_grids.append(grids[sample_id])
        resolved_shapes = [all_shapes[sample_id][index] for index in x_index]
        all_resolved_shapes.append(resolved_shapes)

        for i, resolved_shape in enumerate(resolved_shapes):
            is_original = int(resolved_shape in original_cache[sample_id])
            extra_features[f'shape{i}.is_original'].append(is_original)

    result = generate_df(resolved_grids, all_resolved_shapes)
    return pd.concat([result, pd.DataFrame(extra_features)], axis=1)


def train_model(df: pd.DataFrame, label: np.ndarray,
                params: GlobalParams)->list[MLModel]:
    return make_classifier(df, label, params, 'attention')
