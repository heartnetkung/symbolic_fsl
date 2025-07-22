from abc import ABC, abstractmethod
from ...graphic import *
import pandas as pd
from typing import Optional
from ...constant import *


class ColumnMaker(ABC):
    @abstractmethod
    def append_all(
            self, result: dict[str, list[int]], grids: Optional[list[Grid]],
            all_shapes: Optional[list[list[Shape]]], edit_index: int)->None:
        '''
        Append data to result for DataFrame generation.
        edit_index might be -1.
        '''
        pass


class GridColumns(ColumnMaker):
    def append_all(
            self, result: dict[str, list[int]], grids: Optional[list[Grid]],
            all_shapes: Optional[list[list[Shape]]], edit_index: int)->None:
        if grids is None:
            return

        result['grid_width'] = []
        result['grid_height'] = []
        result['grid_top_color'] = []
        result['grid_second_top_color'] = []
        result['grid_partition_cols'] = []
        result['grid_partition_rows'] = []

        for grid in grids:
            result['grid_width'].append(grid.width)
            result['grid_height'].append(grid.height)
            result['grid_top_color'].append(grid.get_top_color())
            result['grid_second_top_color'].append(grid.get_second_top_color())

            rows, cols, _, _ = grid.separators
            result['grid_partition_cols'].append(len(cols))
            result['grid_partition_rows'].append(len(rows))


class ShapeColumns(ColumnMaker):
    def append_all(
            self, result: dict[str, list[int]], grids: Optional[list[Grid]],
            all_shapes: Optional[list[list[Shape]]], edit_index: int)->None:
        if all_shapes is None:
            return

        for index, shapes in enumerate(all_shapes):
            for i, shape in enumerate(shapes):
                for k, v in shape.to_input_var().items():
                    result_key = f'shape{i}.{k}'
                    result_value = result.get(result_key, None)
                    if result_value is None:
                        result_value = result[result_key] = [MISSING_VALUE]*index
                    result_value.append(v)


class EditColumns(ColumnMaker):
    def append_all(
            self, result: dict[str, list[int]], grids: Optional[list[Grid]],
            all_shapes: Optional[list[list[Shape]]], edit_index: int)->None:
        if (all_shapes is None) or (edit_index == -1):
            return

        result['mass_rank'] = []

        for shapes in all_shapes:
            masses = [shape.mass for shape in shapes]
            result['mass_rank'].append(to_rank(masses)[edit_index])


class ShapeStatsColumns(ColumnMaker):
    def append_all(
            self, result: dict[str, list[int]], grids: Optional[list[Grid]],
            all_shapes: Optional[list[list[Shape]]], edit_index: int)->None:
        if (all_shapes is None) or (len(all_shapes[0]) < 2):
            return

        result['bound_width(shapes)'] = []
        result['bound_height(shapes)'] = []
        result['bound_x(shapes)'] = []
        result['bound_y(shapes)'] = []

        for shapes in all_shapes:
            result['bound_width(shapes)'].append(bound_width(shapes))
            result['bound_height(shapes)'].append(bound_height(shapes))
            result['bound_x(shapes)'].append(bound_x(shapes))
            result['bound_y(shapes)'].append(bound_y(shapes))


def to_rank(values: list[int])->list[int]:
    lookup = {}
    for i, val in enumerate(sorted(set(values), reverse=True)):
        lookup[val] = i
    return [lookup[val] for val in values]
