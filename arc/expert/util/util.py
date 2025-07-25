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


def filter_overwhelming_shapes(all_shapes: list[list[Shape]])-> list[list[Shape]]:
    result = []
    for shapes in all_shapes:
        if len(shapes) > MAX_SHAPES_PER_GRID:
            filtered_shapes = [shape for shape in shapes if shape.mass > 1]
            if len(filtered_shapes) > MAX_SHAPES_PER_GRID:
                filtered_shapes = [shape for shape in shapes if shape.mass > 2]
                if len(filtered_shapes) > MAX_SHAPES_PER_GRID:
                    filtered_shapes = [shape for shape in shapes if shape.mass > 3]
            result.append(filtered_shapes)
        else:
            result.append(shapes)
    return result
