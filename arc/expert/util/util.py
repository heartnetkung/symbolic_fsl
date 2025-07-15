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


def deduplicate_all_shapes(all_shapes: list[list[Shape]])->list[list[Shape]]:
    return [list(dict.fromkeys(shapes)) for shapes in all_shapes]
