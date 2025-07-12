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

    output2 = [Unknown(2*x, 2*y, Grid([[7, 9], [4, 3]]))
               for y, x in product(range(3), range(3))]
    assert output2 == recursive_shape_split(container1, components1, colorless=True)

    output3 = [
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
    assert output3 == recursive_shape_split(container1, components1, transform=True)
