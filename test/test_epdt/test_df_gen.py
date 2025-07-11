from ...arc.ml import generate_df
from ...arc.base import *
from ...arc.graphic import *
import pandas as pd
import numpy as np


def test_df_gen():
    grid = Grid([[1, 2], [3, 4]])
    grid2 = Grid([[1, 2, 3], [3, 4, 5]])
    df = generate_df([grid, grid2])
    assert np.array_equal(df['grid_width'], [2, 3])


def test_df_gen2():
    grid = Grid([[1, 2], [3, 4]])
    grid2 = Grid([[1, 2, 3], [3, 4, 5]])
    extra = {'a': [3, 4], 'z': [1, 2]}
    df = generate_df([grid, grid2], None, extra)
    assert np.array_equal(df['a'], [3, 4])
    assert np.array_equal(df['z'], [1, 2])
    assert np.array_equal(df['grid_width'], [2, 3])


def test_df_gen3():
    grid = Grid([[1, 2], [3, 4]])
    grid2 = Grid([[1, 2, 3], [3, 4, 5]])
    all_shapes = [
        [FilledRectangle(1, 1, 1, 1, 1), FilledRectangle(2, 2, 2, 2, 2)],
        [FilledRectangle(3, 3, 3, 3, 3), FilledRectangle(4, 4, 4, 4, 4)]]
    extra = {'a': [3, 4], 'z': [1, 2]}
    df = generate_df([grid, grid2], all_shapes, extra)
    assert np.array_equal(df['a'], [3, 4])
    assert np.array_equal(df['z'], [1, 2])
    assert np.array_equal(df['shape0.height'], [1, 3])
    assert np.array_equal(df['shape0.mass'], [1, 9])
    assert np.array_equal(df['shape1.height'], [2, 4])
    assert np.array_equal(df['shape1.mass'], [4, 16])
    assert np.array_equal(df['grid_width'], [2, 3])
