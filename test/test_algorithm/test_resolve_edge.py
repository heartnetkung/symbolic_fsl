from ...arc.base import *
from ...arc.graphic import *
from ...arc.algorithm.resolve_edge import *


def test_basic():
    larger1 = Unknown(0, 0, Grid([[-1, 1, -1], [1, 1, 1], [-1, 1, -1]]))
    smaller1 = Unknown(5, 0, Grid([[4, 4, 4], [-1, 4, -1]]))
    grid1 = make_grid(21, 21)
    output1 = Unknown(5, -1, Grid([[-1, 4, -1], [4, 4, 4], [-1, 4, -1]]))
    assert output1 == resolve_edge(smaller1, larger1, grid1, True)

    assert resolve_edge(smaller1, larger1, grid1, False) is None

    smaller3 = Unknown(5, 0, Grid([[4, 5, 4], [-1, 4, -1]]))
    assert resolve_edge(smaller3, larger1, grid1, True) is None

    smaller4 = Unknown(5, 1, Grid([[4, 4, 4], [-1, 4, -1]]))
    assert resolve_edge(smaller4, larger1, grid1, True) is None
