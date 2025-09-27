import pandas as pd
from ...base import *
from typing import Union
from ...manager.task import *
from .util import *
from collections import Counter


TaskWithAtn = Union[AttentionTask, TrainingAttentionTask]


def feat_eng(df: pd.DataFrame, state: ArcState, task: TaskWithAtn,
             edit_index: int)->pd.DataFrame:

    new_fields = {'shape_count': []}
    for out_shapes in state.out_shapes:
        new_fields['shape_count'].append(len(out_shapes))
        _groupby_count(new_fields, out_shapes)

    for key in new_fields:
        if len(new_fields[key]) == len(state.out_shapes):
            df[key] = [new_fields[key][i] for i in task.atn.sample_index]
    return df


def _groupby_count(new_fields: dict[str, list], out_shapes: list[Shape])->None:
    counter = Counter()
    counter.update([f'+count_colorless({colorize(out_shape,1)._grid})'for out_shape in out_shapes])
    counter.update([f'+count({out_shape._grid})' for out_shape in out_shapes])
    for el, count in counter.most_common():
        if el in new_fields:
            new_fields[el].append(count)
        else:
            new_fields[el] = [count]
