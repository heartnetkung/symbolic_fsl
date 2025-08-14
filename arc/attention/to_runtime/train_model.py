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
              sample_index: list[int], all_x_index: list[list[int]])->pd.DataFrame:
    resolved_grids, resolved_shapes = [], []
    for sample_id, x_index in zip(sample_index, all_x_index):
        resolved_grids.append(grids[sample_id])
        resolved_shapes.append([all_shapes[sample_id][index] for index in x_index])
    return generate_df(resolved_grids, resolved_shapes)


def train_model(df: pd.DataFrame, label: np.ndarray,
                params: GlobalParams)->list[MLModel]:
    return make_classifier(df, label, params, 'attention')
