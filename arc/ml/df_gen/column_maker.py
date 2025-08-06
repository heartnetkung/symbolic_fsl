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
            if edit_index < len(masses):
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

        if len(all_shapes[0]) < 2:
            return

        result['inner_bound_width(shapes)'] = []
        result['inner_bound_height(shapes)'] = []
        result['inner_bound_x(shapes)'] = []
        result['inner_bound_y(shapes)'] = []

        for shapes in all_shapes:
            inner_bound = find_inner_bound(shapes[0], shapes[1])
            result['inner_bound_x(shapes)'].append(inner_bound.x)
            result['inner_bound_y(shapes)'].append(inner_bound.y)
            result['inner_bound_width(shapes)'].append(inner_bound.width)
            result['inner_bound_height(shapes)'].append(inner_bound.height)


def to_rank(values: list[int], reverse: bool = True)->list[int]:
    lookup = {}
    for i, val in enumerate(sorted(set(values), reverse=reverse)):
        lookup[val] = i
    return [lookup[val] for val in values]
