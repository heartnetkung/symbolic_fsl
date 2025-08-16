from ...arc.base import *
from ...arc.graphic import *
from ...arc.algorithm.recursive_shape_split import *
from itertools import product


def test_basic():
    container1 = Unknown(0, 0, Grid([
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [9, 7, 9, 7, 9, 7],
        [3, 4, 3, 4, 3, 4],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3]]))
    components1 = [Unknown(0, 0, Grid([[7, 9], [4, 3]]))]
    assert recursive_shape_split(container1, components1) is None

    output2 = [
        Unknown(0, 0, Grid([[7, 9], [4, 3]])),
        Unknown(2, 0, Grid([[7, 9], [4, 3]])),
        Unknown(4, 0, Grid([[7, 9], [4, 3]])),
        Unknown(0, 2, Grid([[9, 7], [3, 4]])),
        Unknown(2, 2, Grid([[9, 7], [3, 4]])),
        Unknown(4, 2, Grid([[9, 7], [3, 4]])),
        Unknown(0, 4, Grid([[7, 9], [4, 3]])),
        Unknown(2, 4, Grid([[7, 9], [4, 3]])),
        Unknown(4, 4, Grid([[7, 9], [4, 3]]))
    ]
    assert output2 == recursive_shape_split(container1, components1, colorless=True)
    assert output2 == recursive_shape_split(container1, components1, transform=True)


def test_98():
    subshapes = [Unknown(0, 0, y98_subshape),
                 Unknown(0, 0, y98_subshape2), Unknown(0, 0, y98_subshape3)]
    output = recursive_shape_split(Unknown(1, 1, y98), subshapes, colorless=True)
    assert output == [
        Unknown(1, 1, Grid([[2, 2, 2, 2, 2],
                            [1, 1, 2, 1, 1],
                            [1, 2, 2, 2, 1],
                            [1, 2, 2, 2, 1],
                            [1, 1, 1, 1, 1]])),
        Unknown(4, 6, Grid([[3, 3, 3, 3, 3],
                            [1, 1, 3, 1, 1],
                            [1, 3, 3, 3, 1],
                            [1, 1, 1, 1, 1]]))
    ]


y98_subshape = Grid([
    [2, 2, 2, 2, 2],
    [1, 1, 2, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1]
])
y98_subshape2 = Grid([
    [6, 6, 6, 6, 6],
    [1, 1, 6, 1, 1],
    [1, 6, 6, 6, 1],
    [1, 6, 6, 6, 1],
    [1, 1, 1, 1, 1]
])
y98_subshape3 = Grid([
    [8, 8, 8, 8, 8],
    [1, 1, 8, 1, 1],
    [1, 8, 8, 8, 1],
    [1, 1, 1, 1, 1]
])

y98 = Grid([
    [2, 2, 2, 2, 2, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, -1, -1],
    [1, 2, 2, 2, 1, -1, -1, -1],
    [1, 2, 2, 2, 1, -1, -1, -1],
    [1, 1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 3, 3, 3, 3, 3],
    [-1, -1, -1, 1, 1, 3, 1, 1],
    [-1, -1, -1, 1, 3, 3, 3, 1],
    [-1, -1, -1, 1, 1, 1, 1, 1]
])
