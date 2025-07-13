from ...arc.base import *
from ...arc.graphic import *
from ...arc.attention.low_level import *


def test_properties():
    rect1, rect2 = FilledRectangle(0, 0, 1, 1, 2), FilledRectangle(1, 1, 1, 1, 2)
    relationships = list_relationship(rect1, rect2)
    assert 'same_height' in relationships
    assert 'same_width' in relationships
    assert 'same_top_color' in relationships
    assert 'same_mass' in relationships

    unknown1 = Unknown(0, 0, Grid([[1, 2], [3, 4]]))
    rotated_unknown1 = Unknown(0, 0, Grid([[2, 4], [1, 3]]))
    relationships = list_relationship(unknown1, rotated_unknown1)
    assert 'same_transformed_shape' in relationships
    assert 'same_colorless_transformed_shape' in relationships

    container = Unknown(0, 0, Grid(
        [[-1, 3, -1, -1], [3, -1, 3, -1], [-1, 3, -1, 3], [-1, -1, 3, -1]]))
    contained = FilledRectangle(1, 1, 1, 1, 4)
    relationships = list_relationship(container, contained)
    assert 'contain' in relationships

    rect4, rect5 = FilledRectangle(0, 0, 2, 2, 1), FilledRectangle(1, 1, 2, 2, 1)
    relationships = list_relationship(rect4, rect5)
    assert 'overlap' in relationships
    assert 'touch' not in relationships

    rect6 = FilledRectangle(1, 2, 2, 2, 1)
    relationships = list_relationship(rect4, rect6)
    assert 'touch' in relationships
    assert 'overlap' not in relationships


def test_exact_contain():
    larger_shape = Unknown(0, 0, Grid([
        [1, 1, 1, 1], [1, -1, 1, 1], [1, -1, -1, 1], [1, 1, 1, 1]]))
    smaller_shape1 = Unknown(0, 0, Grid([[1, -1], [1, 1]]))
    relationships = list_relationship(larger_shape, smaller_shape1)
    assert 'exact_contain' in relationships
    assert 'exact_transformed_contain' in relationships

    smaller_shape2 = Unknown(0, 0, Grid([[-1, 1], [1, 1]]))
    relationships = list_relationship(larger_shape, smaller_shape2)
    assert 'exact_contain' not in relationships
    assert 'exact_transformed_contain' in relationships


def test_subshape():
    unknown_17_1 = Unknown(0, 0, Grid([[-1, 8, -1], [3, 8, 1], [8, 4, 8]]))
    unknown_17_2 = Unknown(0, 0, Grid([[1, -1], [-1, 4], [3, -1]]))
    relationships = list_relationship(unknown_17_1, unknown_17_2)
    assert 'transformed_subshape' in relationships
    assert 'subshape' not in relationships

    unknown_17_3 = Unknown(0, 0, Grid([
        [-1, -1, 4, -1, -1], [-1, -1, -1, -1, -1], [3, -1, -1, -1, 1]]))
    relationships = list_relationship(unknown_17_1, unknown_17_3)
    assert 'transformed_subshape' not in relationships
    assert 'subshape' not in relationships

    unknown_21_1 = Unknown(0, 0, Grid([[6, 6, 7], [-1, 5, 7], [4, 4, -1]]))
    unknown_21_2 = Unknown(0, 0, Grid([[-1, 7], [5, 7]]))
    relationships = list_relationship(unknown_21_1, unknown_21_2)
    assert 'transformed_subshape' in relationships
    assert 'subshape' in relationships

    unknown_75_1 = Unknown(0, 0, Grid([
        [-1, -1, 2, -1], [4, 4, 4, 1], [-1, -1, 4, -1], [3, 3, 4, 3]]))
    unknown_75_2 = Unknown(0, 0, Grid([
        [2, -1, -1], [4, 4, 4], [4, -1, -1], [4, -1, -1]]))
    relationships = list_relationship(unknown_75_1, unknown_75_2)
    assert 'transformed_subshape' in relationships
    assert 'subshape' not in relationships


def test_scale():
    unknown_100_1 = Unknown(0, 0, Grid([
        [1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, 2, 2],
        [1, 1, -1, -1, 2, 2],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]]))
    unknown_100_2 = Unknown(0, 0, Grid([
        [1, 1, 1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, -1, -1, -1, 2, 2, 2],
        [1, 1, 1, -1, -1, -1, 2, 2, 2],
        [1, 1, 1, -1, -1, -1, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    relationships = list_relationship(unknown_100_1, unknown_100_2)
    assert 'unknown_scale' in relationships
