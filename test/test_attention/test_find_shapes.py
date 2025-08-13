from ...arc.base import *
from ...arc.graphic import *
from ...arc.attention.low_level import *
from ...arc.attention.make_attention.find_shapes import *


def test_281():
    all_y_shapes = [
        list_sparse_objects(y281_0.replace_color(0, NULL_COLOR)),
        list_sparse_objects(y281_1.replace_color(0, NULL_COLOR))
    ]
    result = find_common_y_shapes(all_y_shapes)
    assert result == [Unknown(0, 0, Grid([[5, 1, 5], [1, -1, 1], [5, 1, 5]]))]


def test_333():
    all_y_shapes = [
        [Unknown(0, 0, y333_0)],
        [Unknown(0, 0, y333_1)],
        [Unknown(0, 0, y333_2)],
        [Unknown(0, 0, y333_3)],
        [Unknown(0, 0, y333_4)],
        [Unknown(0, 0, y333_5)],
        [Unknown(0, 0, y333_6)]
    ]
    result = find_common_y_shapes(all_y_shapes)
    assert set(result) == {shapes[0] for shapes in all_y_shapes}


def test_370():

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    all_y_shapes = [
        list_sparse_objects(y370_0.replace_color(0, NULL_COLOR)),
        list_sparse_objects(y370_1.replace_color(0, NULL_COLOR))
    ]
    result = find_common_y_shapes(all_y_shapes)
    expected = {Unknown(0, 0, Grid([[-1, 3, -1], [3, 3, 3], [-1, 3, -1]]))}
    assert set(result) == expected


def test_398():
    all_y_shapes = [
        [Unknown(0, 0, y398_0.replace_color(0, NULL_COLOR)),
         Unknown(0, 0, y398_0.replace_color(0, NULL_COLOR))],
        [Unknown(0, 0, y398_1.replace_color(0, NULL_COLOR))],
        [Unknown(0, 0, y398_2.replace_color(0, NULL_COLOR))],
        [Unknown(0, 0, y398_3.replace_color(0, NULL_COLOR))],
        [Unknown(0, 0, y398_4.replace_color(0, NULL_COLOR))],
        [Unknown(0, 0, y398_5.replace_color(0, NULL_COLOR))],
        [Unknown(0, 0, y398_6.replace_color(0, NULL_COLOR))],
        [Unknown(0, 0, y398_7.replace_color(0, NULL_COLOR))]
    ]
    result = find_common_y_shapes(all_y_shapes)
    assert set(result) == {shapes[0] for shapes in all_y_shapes}


y281_0 = Grid([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 1, 5, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 5, 1, 5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 1, 5, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 5, 1, 5, 0, 5, 1, 5, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 5, 1, 5, 0, 0, 0, 0, 0]
])

y281_1 = Grid([
    [0, 5, 1, 5, 0, 0, 5, 1, 5],
    [0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 5, 1, 5, 0, 0, 5, 1, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 1, 5, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 5, 1, 5, 0, 5, 1, 5, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 5, 1, 5, 0]
])

y333_0 = Grid([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0]
])

y333_1 = Grid([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0]
])

y333_2 = Grid([
    [0, 0, 5],
    [0, 0, 5],
    [5, 5, 5]
])

y333_3 = Grid([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0]
])

y333_4 = Grid([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0]
])

y333_5 = Grid([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0]
])

y333_6 = Grid([
    [0, 0, 5],
    [0, 0, 5],
    [5, 5, 5]
])

y370_0 = Grid([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 1, 0, 3, 3, 3, 0, 1, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

y370_1 = Grid([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 3, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

y398_0 = Grid([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

y398_1 = Grid([
    [1, 0, 1],
    [0, 0, 0],
    [0, 0, 0]
])

y398_2 = Grid([
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
])

y398_3 = Grid([
    [1, 0, 1],
    [0, 0, 0],
    [0, 0, 0]
])

y398_4 = Grid([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

y398_5 = Grid([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])

y398_6 = Grid([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

y398_7 = Grid([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])
