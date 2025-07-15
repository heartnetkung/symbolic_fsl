from ...graphic import *
from ...ml import *
import pandas as pd
import numpy as np
from ...base import *


def make_single_shape_df(state: ArcState)->pd.DataFrame:
    assert state.out_shapes != None
    assert len(state.out_shapes) == len(state.x)

    grids, all_shapes = [], []
    for grid, shapes in zip(state.x, state.out_shapes):
        for shape in shapes:
            grids.append(grid)
            all_shapes.append([shape])
    return generate_df(grids, all_shapes)
