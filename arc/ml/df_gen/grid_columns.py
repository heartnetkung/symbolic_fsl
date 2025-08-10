from ...graphic import *
from functools import lru_cache
from .column_maker import ColumnMaker
from typing import Optional
from ...constant import MISSING_VALUE


class GridColumns(ColumnMaker):
    def append_all(
            self, result: dict[str, list[float]], grids: Optional[list[Grid]],
            all_shapes: Optional[list[list[Shape]]], edit_index: int)->None:
        if grids is None:
            return

        result['grid_width'] = []
        result['grid_height'] = []
        result['grid_top_color'] = []
        result['grid_second_top_color'] = []
        result['grid_least_top_color'] = []
        result['grid_partition_cols'] = []
        result['grid_partition_rows'] = []
        result['grid_largest_color'] = []

        for grid in grids:
            result['grid_width'].append(grid.width)
            result['grid_height'].append(grid.height)
            result['grid_top_color'].append(grid.get_top_color())
            result['grid_second_top_color'].append(grid.get_second_top_color())
            result['grid_least_top_color'].append(grid.get_least_color())

            for k, v in grid_feat_eng(grid).items():
                result[k].append(v)


@lru_cache
def grid_feat_eng(grid: Grid)->dict[str, int]:
    result = {}

    rows, cols, _, _ = find_separators(grid)
    result['grid_partition_cols'] = len(cols)
    result['grid_partition_rows'] = len(rows)

    shapes = list_objects(grid)
    if len(shapes) == 0:
        result['grid_largest_color'] = MISSING_VALUE
    else:
        result['grid_largest_color'] = max(
            shapes, key=lambda shape: shape.mass).top_color

    return result
