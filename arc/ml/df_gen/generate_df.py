from ...graphic import *
import pandas as pd
from typing import Optional
from ...constant import *
from .edit_feat_eng import edit_feat_eng


def generate_df(
        grids: list[Grid], all_shapes: Optional[list[list[Shape]]] = None,
        extra_features: Optional[dict[str, list]] = None,
        edit_index: int = -1)->pd.DataFrame:
    '''
    Usual routine of generating dataframe from shapes and grids.
    '''
    _check_input(grids, all_shapes, extra_features)
    extra_features_sub = extra_features if extra_features is not None else {}
    result = {'grid_width': [], 'grid_height': []} | extra_features_sub

    for grid in grids:
        _append_grid(grid, result)
    if all_shapes is not None:
        for i, shapes in enumerate(all_shapes):
            _append_shapes(shapes, result, i)
        if edit_index != -1:
            edit_feat_eng(all_shapes, edit_index, result)

    return pd.DataFrame(_sort_key(ensure_size(result)))


def ensure_size(df_dict: dict[str, list])->dict[str, list]:
    n_rows = max([len(values) for values in df_dict.values()])
    for key, values in df_dict.items():
        len_diff = n_rows-len(values)
        if len_diff != 0:
            df_dict[key] = values+([MISSING_VALUE]*len_diff)
    return df_dict


def _check_input(grids: list[Grid],
                 all_shapes: Optional[list[list[Shape]]],
                 extra_features: Optional[dict[str, list]])->None:
    len_grids = len(grids)
    assert len_grids > 0

    if extra_features is not None:
        for k, v in extra_features.items():
            assert len(v) == len_grids

    if all_shapes is not None:
        assert len(all_shapes) > 0
        assert len(all_shapes) == len_grids
        all_shapes_count = {len(shapes) for shapes in all_shapes}
        assert len(all_shapes_count) == 1
        assert all_shapes_count.pop() > 0


def _append_grid(grid: Grid, result: dict[str, list])->None:
    result['grid_width'].append(grid.width)
    result['grid_height'].append(grid.height)


def _append_shapes(shapes: list[Shape], result: dict[str, list], index: int)->None:
    for i, shape in enumerate(shapes):
        for k, v in shape.to_input_var().items():
            result_key = f'shape{i}.{k}'
            result_value = result.get(result_key, None)
            if result_value is None:
                result_value = result[result_key] = [MISSING_VALUE]*index
            result_value.append(v)


def _sort_key(data: dict[str, list])->dict[str, list]:
    keys = sorted(data.keys(), key=lambda x: ' '+x if x.startswith('shape') else x)
    return {key: data[key] for key in keys}
