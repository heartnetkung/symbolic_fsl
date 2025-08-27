from ...arc.graphic import *
import numpy as np


def test_measure_gap():
    shape1 = Unknown(0, 0, Grid([[1, 1, -1, 1, 1],
                                 [1, -1, -1, -1, 1],
                                 [1, -1, -1, -1, 1],
                                 [1, -1, -1, -1, 1],
                                 [1, 1, 1, 1, 1]]))
    shape2 = FilledRectangle(2, 2, 1, 1, 2)
    assert np.isclose(measure_gap(shape1, shape2, 99), 2)
    assert np.isclose(measure_gap(shape2, shape1, 99), 2)

    shape3 = FilledRectangle(3, 3, 1, 1, 2)
    assert np.isclose(measure_gap(shape2, shape3, 99), np.sqrt(2))
    assert np.isclose(measure_gap(shape3, shape2, 99), np.sqrt(2))

    shape4 = FilledRectangle(4, 4, 1, 1, 2)
    assert np.isclose(measure_gap(shape2, shape4, 99), np.sqrt(8))
    assert np.isclose(measure_gap(shape4, shape2, 99), np.sqrt(8))

    assert np.isclose(measure_gap(shape1, shape1, 99), 0)
    assert np.isclose(measure_gap(shape2, shape2, 99), 0)
    assert np.isclose(measure_gap(shape3, shape3, 99), 0)
