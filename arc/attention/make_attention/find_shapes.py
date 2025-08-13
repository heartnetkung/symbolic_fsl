from ...graphic import *
from typing import Optional
import pandas as pd
from copy import deepcopy


def find_common_y_shapes(all_y_shapes: list[list[Shape]])->list[Shape]:
    return _find_common_shapes(all_y_shapes)


def _find_common_shapes(all_y_shapes: list[list[Shape]])->list[Shape]:
    stats = {'sample_id': [], 'repr_': []}
    shape_dict = {}

    for sample, y_shapes in enumerate(all_y_shapes):
        for index, y_shape in enumerate(y_shapes):
            shape_repr = _to_repr(y_shape._grid)
            if shape_repr is None:
                continue

            stats['sample_id'].append(sample)
            stats['repr_'].append(shape_repr)
            shape_dict[shape_repr] = _reset_position(y_shape)

    stats_df = pd.DataFrame(stats)
    grouped_df = stats_df.groupby('repr_').agg({'sample_id': 'nunique'})
    filtered_df = grouped_df[grouped_df['sample_id'] > 1]
    assert isinstance(filtered_df, pd.DataFrame)

    # too many repeated shapes, include everything
    if len(filtered_df) >= 3:
        return list(shape_dict.values())

    return [shape_dict[key] for key in filtered_df.index]


def _to_repr(grid: Grid)->Optional[str]:
    if grid.width*grid.height <= 2:
        return None
    return repr(grid.normalize_color())


def _reset_position(shape: Shape)->Shape:
    if shape.x == shape.y == 0:
        return shape

    result = deepcopy(shape)
    result.x = 0
    result.y = 0
    return result
