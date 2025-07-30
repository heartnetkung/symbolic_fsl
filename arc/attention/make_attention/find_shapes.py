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
