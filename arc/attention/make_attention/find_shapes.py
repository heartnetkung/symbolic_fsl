from ...graphic import *
from ..low_level import *
from typing import Optional


def find_common_y_shapes(all_y_shapes: list[list[Shape]])->list[Shape]:
    props = gen_prop(all_y_shapes, False)
    common_shapes = props.groupby('prop').aggregate('first').drop_duplicates()
    result = []
    for index, row in common_shapes.iterrows():
        sample_id, shape_index = int(row['sample_id']), int(row['index'])
        result.append(ShapeId(sample_id, shape_index))
    return [lookup(shape_id, all_y_shapes)for shape_id in sorted(result)]


def find_syntactic_x_shapes(all_X_train_shapes: list[list[Shape]],
                            all_X_test_shapes: list[list[Shape]])->Optional[tuple[
        dict[int, ShapeId], dict[int, ShapeId]]]:

    len_X_train, len_X_test = len(all_X_train_shapes), len(all_X_test_shapes)
    props = gen_prop(all_X_train_shapes)
    grouping = props.groupby('prop').aggregate({'sample_id': ['count', 'nunique']})
    filter1 = grouping['sample_id', 'count'] == len_X_train
    filter2 = grouping['sample_id', 'nunique'] == len_X_train
    used_prop = grouping[np.logical_and(filter1, filter2)]
    assert isinstance(used_prop, pd.DataFrame)
    if len(used_prop) == 0:
        return None

    to_join = pd.DataFrame({'prop': used_prop.index})
    filtered_train = props.merge(to_join, on='prop', how='inner')[
        ['sample_id', 'index']].drop_duplicates()
    assert isinstance(filtered_train, pd.DataFrame)
    if filtered_train.empty:
        return None

    props2 = gen_prop(all_X_test_shapes)
    grouping2 = props2.merge(to_join, on='prop', how='inner').groupby(
        ['sample_id', 'index']).size().reset_index()
    filtered_test = grouping2.sort_values(['sample_id', 0], ascending=False).groupby(
        'sample_id').aggregate({'index': 'first'}).reset_index()
    return _to_dict(filtered_train), _to_dict(filtered_test)


def _to_dict(df: pd.DataFrame)->dict[int, ShapeId]:
    result = {}
    for i, row in df.iterrows():
        sample_id, shape_index = int(row['sample_id']), int(row['index'])
        result[sample_id] = ShapeId(sample_id, shape_index)
    return result
