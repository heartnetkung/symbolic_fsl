from ...graphic import *
from ..low_level import *
from ...ml import *
import numpy as np
import pandas as pd
from ...global_attention import *


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
              original_shapes: list[list[Shape]],
              g_atn: Optional[GlobalAttention])->pd.DataFrame:
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

    extra_columns: list[ColumnMaker] = []
    if g_atn is not None:
        extra_columns = [_QueryShapeColumn(query, i, original_shapes, sample_index)
                         for i, query in enumerate(g_atn.shape_queries)]

    result = generate_df(
        resolved_grids, all_resolved_shapes, extra_columns=extra_columns)
    return pd.concat([result, pd.DataFrame(extra_features)], axis=1)


def train_model(df: pd.DataFrame, label: np.ndarray,
                params: GlobalParams)->list[MLModel]:
    return make_classifier(df, label, params, 'attention')


class _QueryShapeColumn(ColumnMaker):
    def __init__(self, query: ShapeQuery, index: int,
                 all_full_shapes: list[list[Shape]], sample_index: list[int])->None:
        self.query = query
        self.index = index
        self.all_full_shapes = all_full_shapes
        self.sample_index = sample_index

    def append_all(
            self, result: dict[str, list[float]], grids: Optional[list[Grid]],
            all_shapes: Optional[list[list[Shape]]], edit_index: int)->None:
        assert all_shapes is not None
        if self.query.is_null():
            return

        for index, (id1, shapes) in enumerate(zip(self.sample_index, all_shapes)):
            id2 = self.query.shape_index[id1]
            query_shape = self.all_full_shapes[id1][id2]
            shape_properties = query_shape.to_input_var() | _make_extra_column(
                query_shape, shapes)
            for k, v in shape_properties.items():
                result_key = f'+query{self.index}.{k}'
                result_value = result.get(result_key, None)
                if result_value is None:
                    result_value = result[result_key] = [MISSING_VALUE]*index
                result_value.append(v)


def _make_extra_column(query_shape: Shape, column_shapes: list[Shape])->dict[str, int]:
    result = {}
    for i, shape in enumerate(column_shapes):
        key = f'same_shape(shape{i})'
        if query_shape == shape:
            result[key] = 0
        else:
            result[key] = int(query_shape.shape_value == shape.shape_value)

        result[f'is_contain(shape{i})'] = int(len(is_contain(query_shape, shape)) > 0)
    return result
