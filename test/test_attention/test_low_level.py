from ...arc.base import *
from ...arc.graphic import *
from ...arc.attention.low_level import *
import pandas as pd


def test_gen_rel_product():
    all_x_shapes = [[FilledRectangle(0, 0, 1, 1, 1), FilledRectangle(0, 0, 1, 3, 2)],
                    [FilledRectangle(0, 0, 2, 2, 1)]]
    all_y_shapes = [[FilledRectangle(0, 0, 1, 1, 2)], [FilledRectangle(0, 0, 2, 2, 2)]]
    result = gen_rel_product(all_x_shapes, all_y_shapes)
    expect = {'same_mass', 'same_center', 'same_colorless_transformed_shape',
              'same_x', 'same_colorless_shape', 'same_height', 'same_y', 'same_width'}
    assert expect.issubset(set(result['rel']))


def test_gen_prop():
    all_shapes = [
        [FilledRectangle(1, 1, 1, 1, 1), FilledRectangle(2, 2, 2, 2, 2)],
        [FilledRectangle(3, 3, 3, 3, 1)]]
    expect = {'sample_id': [0, 0, 1, 1],
              'index': [0, 0, 0, 0],
              'prop': ['top_color: 1', 'colors: {1}', 'top_color: 1', 'colors: {1}']}
    result = gen_prop(all_shapes)
    assert pd.DataFrame(expect).equals(result)

    all_shapes2 = [
        [FilledRectangle(1, 1, 1, 1, 1), FilledRectangle(2, 2, 2, 2, 2)],
        [FilledRectangle(3, 3, 3, 3, 3)]]
    result2 = gen_prop(all_shapes2)
    assert result2.empty


def test_constant_arity():
    input_ = {'sample': [0, 0, 1, 1, 1, 1],
              'rel': ['same_x', 'same_y', 'same_x', 'same_y', 'same_y', 'same_z']}
    expect = {'sample': [0, 1],
              'rel': ['same_x', 'same_x']}
    result = filter_constant_arity(pd.DataFrame(input_), ['sample'], 'rel')
    assert pd.DataFrame(expect).equals(result)
