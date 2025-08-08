from ...arc.base import *
from ...arc.graphic import *
from ...arc.algorithm.find_fitb_sym_bound import *


def test_largest_symmetry():
    mode = SymmetryMode.double_mirror
    result = find_largest_symmetry(x116_0)
    assert result == (3, 4, 3, 3)
    assert cal_width_height(x116_0, result, mode) == (9, 11)
    assert cal_offset(x116_0, result, mode) == (0, 0)

    result = find_largest_symmetry(x116_1)
    assert result == (0, 3, 3, 3)
    assert cal_width_height(x116_1, result, mode) == (9, 9)
    assert cal_offset(x116_1, result, mode) == (3, 0)

    result = find_largest_symmetry(x116_2)
    assert result == (3, 0, 3, 3)
    assert cal_width_height(x116_2, result, mode) == (9, 9)
    assert cal_offset(x116_2, result, mode) == (0, 3)

    result = find_largest_symmetry(x116_3)
    assert result == (0, 0, 3, 3)
    assert cal_width_height(x116_3, result, mode) == (9, 13)
    assert cal_offset(x116_3, result, mode) == (3, 5)

    mode = SymmetryMode.full_rotation
    result = find_largest_symmetry(x360_0)
    assert result == (0, 1, 3, 3)
    assert cal_width_height(x360_0, result, mode) == (5, 5)
    assert cal_offset(x360_0, result, mode) == (1, 0)

    result = find_largest_symmetry(x360_1)
    assert result == (1, 1, 2, 2)
    assert cal_width_height(x360_1, result, mode) == (4, 4)
    assert cal_offset(x360_1, result, mode) == (0, 0)

    result = find_largest_symmetry(x360_2)
    assert result == (0, 1, 3, 3)
    assert cal_width_height(x360_2, result, mode) == (5, 5)
    assert cal_offset(x360_2, result, mode) == (1, 0)

    result = find_largest_symmetry(x360_3)
    assert result == (2, 1, 3, 3)
    assert cal_width_height(x360_3, result, mode) == (7, 7)
    assert cal_offset(x360_3, result, mode) == (0, 1)


x116_0 = Grid([
    [2, -1, -1, 2, -1, -1],
    [2, 2, -1, 2, -1, -1],
    [-1, -1, 2, 2, -1, -1],
    [-1, 2, 2, -1, -1, -1],
    [2, -1, -1, 4, -1, 4],
    [-1, -1, -1, -1, 4, -1],
    [-1, -1, -1, 4, -1, 4]
])
x116_1 = Grid([
    [-1, -1, -1, -1, 8, -1],
    [-1, -1, -1, 8, 8, 8],
    [-1, -1, 8, 8, 8, -1],
    [3, -1, 3, -1, -1, -1],
    [-1, 3, -1, -1, -1, -1],
    [3, -1, 3, -1, -1, -1],
])
x116_2 = Grid([
    [-1, -1, -1, 8, -1, 8],
    [-1, -1, -1, -1, 8, -1],
    [-1, -1, -1, 8, -1, 8],
    [-1, 1, 1, -1, -1, -1],
    [1, -1, 1, -1, -1, -1],
    [-1, 1, -1, -1, -1, -1]
])
x116_3 = Grid([
    [7, -1, 7, -1, -1, -1],
    [-1, 7, -1, -1, -1, -1],
    [7, -1, 7, -1, 4, -1],
    [-1, -1, 4, 4, -1, 4],
    [-1, -1, -1, 4, -1, -1],
    [-1, -1, -1, 4, 4, -1],
    [-1, -1, 4, -1, -1, -1],
    [-1, -1, 4, -1, -1, -1]
])

x360_0 = Grid([
    [-1, 7, -1, -1],
    [4, 7, 4, -1],
    [7, 4, 7, -1],
    [4, 7, 4, -1],
    [-1, -1, -1, 4],
])
x360_1 = Grid([
    [3, -1, -1],
    [-1, 6, 6],
    [-1, 6, 6],
    [-1, -1, 6],
])
x360_2 = Grid([
    [-1, -1, -1, 9],
    [8, 8, 8, -1],
    [8, 8, 8, -1],
    [8, 8, 8, -1],
])
x360_3 = Grid([
    [-1, -1, -1, 2, -1],
    [3, 3, 3, 2, 3],
    [-1, -1, 2, 3, 2],
    [3, 3, 3, 2, 3],
])
