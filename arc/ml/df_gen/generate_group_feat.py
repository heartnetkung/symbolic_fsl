import pandas as pd
from ...base import *
from ...graphic import *
from typing import Union
from collections import Counter
from .column_maker import to_rank_float


def gen_group_feat(df: pd.DataFrame, all_shapes: list[list[Shape]],
                   sample_index: list[int], all_x_index: list[list[int]],
                   edit_index: int = -1)->pd.DataFrame:

    new_fields = {'shape_count': []}
    for shapes in all_shapes:
        new_fields['shape_count'].append(len(shapes))
        _groupby_count(new_fields, shapes)

    for key in new_fields:
        if len(new_fields[key]) == len(all_shapes):
            df[key] = [new_fields[key][i] for i in sample_index]

    if edit_index > -1:
        try:
            mass_rank = []
            for sample_id, x_index in zip(sample_index, all_x_index):
                masses = [shape.mass for shape in all_shapes[sample_id]]
                mass_rank.append(to_rank_float(masses)[x_index[edit_index]])
            df['mass_rank'] = mass_rank
        except Exception:
            pass

    return df


def _groupby_count(new_fields: dict[str, list], shapes: list[Shape])->None:
    counter = Counter()
    counter.update([f'+count_colorless({colorize(shape,1)._grid})'for shape in shapes])
    counter.update([f'+count({shape._grid})' for shape in shapes])
    for el, count in counter.most_common():
        if el in new_fields:
            new_fields[el].append(count)
        else:
            new_fields[el] = [count]
