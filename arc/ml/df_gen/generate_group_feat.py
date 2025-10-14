import pandas as pd
from ...base import *
from ...graphic import *
from typing import Union
from collections import Counter
from .column_maker import to_rank_float


def gen_group_feat(df: pd.DataFrame, all_shapes: list[list[Shape]],
                   sample_index: list[int], all_x_index: list[list[int]],
                   edit_index: int = -1)->pd.DataFrame:
    try:
        new_fields = {'shape_count': []}
        for shapes in all_shapes:
            new_fields['shape_count'].append(len(shapes))
            _make_grid_oriented_field(new_fields, shapes)

        for key in new_fields:
            if len(new_fields[key]) == len(all_shapes):
                df[key] = [new_fields[key][i] for i in sample_index]

        if edit_index > -1:
            _make_shape_oriented_field(
                df, all_shapes, sample_index, all_x_index, edit_index)
    except Exception:
        pass
    return df


def _make_grid_oriented_field(new_fields: dict[str, list], shapes: list[Shape])->None:
    counter = Counter()
    counter.update([f'+count_colorless({colorize(shape,1)._grid})'for shape in shapes])
    counter.update([f'+count({shape._grid})' for shape in shapes])
    for el, count in counter.most_common():
        if el in new_fields:
            new_fields[el].append(count)
        else:
            new_fields[el] = [count]


def _make_shape_oriented_field(
        df: pd.DataFrame, all_shapes: list[list[Shape]], sample_index: list[int],
        all_x_index: list[list[int]], edit_index: int = -1)->None:
    _make_mass_rank(df, all_shapes, sample_index, all_x_index)
    _make_color_rank(df, all_shapes, sample_index, all_x_index)


def _make_mass_rank(
        df: pd.DataFrame, all_shapes: list[list[Shape]], sample_index: list[int],
        all_x_index: list[list[int]], edit_index: int = -1)->None:
    mass_rank_cache = {i: [shape.mass for shape in all_shapes[i]]
                       for i in range(len(all_shapes))}
    column = []
    for sample_id, x_index in zip(sample_index, all_x_index):
        masses = mass_rank_cache[sample_id]
        column.append(to_rank_float(masses)[x_index[edit_index]])
    df['mass_rank'] = column


def _make_color_rank(
        df: pd.DataFrame, all_shapes: list[list[Shape]], sample_index: list[int],
        all_x_index: list[list[int]], edit_index: int = -1)->None:
    for color in range(1, 10):
        ranks, column = {}, []
        for sample_id in range(len(all_shapes)):
            ranks[sample_id] = [
                shape._grid.color_count[color] for shape in all_shapes[sample_id]]

        for sample_id, x_index in zip(sample_index, all_x_index):
            rank = ranks[sample_id]
            column.append(to_rank_float(rank)[x_index[edit_index]])
        df[f'+color_{color}_rank'] = column
