from ...graphic import *
import pandas as pd
from typing import Optional
from ...constant import *
from .column_maker import *
from .grid_columns import *


def generate_df(
        grids: Optional[list[Grid]] = None,
        all_shapes: Optional[list[list[Shape]]] = None,
        extra_columns: Optional[list[ColumnMaker]] = None,
        edit_index: int = -1)->pd.DataFrame:
    '''
    Usual routine of generating dataframe from shapes and grids.
    '''
    _check_input(grids, all_shapes)
    result = {}
    columns = [GridColumns(), ShapeColumns(), EditColumns(), ShapeStatsColumns()]
    if extra_columns is not None:
        columns += extra_columns

    for col in columns:
        col.append_all(result, grids, all_shapes, edit_index)
    return pd.DataFrame(_sort_key(ensure_size(result)))


def ensure_size(df_dict: dict[str, list])->dict[str, list]:
    n_rows = max([len(values) for values in df_dict.values()])
    for key, values in df_dict.items():
        len_diff = n_rows-len(values)
        if len_diff != 0:
            df_dict[key] = values+([MISSING_VALUE]*len_diff)
    return df_dict


def _check_input(grids: Optional[list[Grid]],
                 all_shapes: Optional[list[list[Shape]]])->None:
    if grids is not None:
        _len = len(grids)
    elif all_shapes is not None:
        _len = len(all_shapes)
    else:
        assert False
    assert _len > 0

    if all_shapes is not None:
        assert len(all_shapes) == _len
        all_shapes_count = {len(shapes) for shapes in all_shapes}
        assert len(all_shapes_count) == 1


def _sort_key(data: dict[str, list])->dict[str, list]:
    keys = sorted(data.keys(), key=lambda x: ' '+x if x.startswith('shape') else x)
    return {key: data[key] for key in keys}
