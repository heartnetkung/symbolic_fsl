from ...arc.ml.linprog import *
from ...arc.base import *
import pandas as pd
import numpy as np
import re

X = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]})


def test_solve_all():
    params = GlobalParams(ppdt_enable_deg2=True)
    y = np.array([0, 1, 2])
    result = solve_reg(X, y, params)
    assert len(result) == 2
    assert np.allclose(result[0].poly_coef, [0, 1, 0, 0, 0, 0])
    assert np.allclose(result[1].poly_coef, [-3, 0, 1, 0, 0, 0])

    y = np.array([0, 0, 0])
    result = solve_reg(X, y, params)
    assert len(result) == 1
    assert np.allclose(result[0].poly_coef, [0, 0, 0, 0, 0, 0])

    y = np.array([5, 5, 5])
    result = solve_reg(X, y, params)
    assert len(result) == 1
    assert np.allclose(result[0].poly_coef, [5, 0, 0, 0, 0, 0])

    # overfit
    y = np.array([18, 32, 50])
    result = solve_reg(X, y, params)
    assert len(result) == 0

    X2 = pd.DataFrame({'a': [0, 1, 2, 3], 'b': [3, 4, 5, 6]})
    y = np.array([18, 32, 50, 72])
    result = solve_reg(X2, y, params)
    assert len(result) == 5
    assert np.allclose(result[0].poly_coef, [0, 0, 0, 0, 0, 2])
    assert np.allclose(result[1].poly_coef, [0, 0, 6, 0, 2, 0])
    assert np.allclose(result[2].poly_coef, [18, 0, 0, -2, 4, 0])
    assert np.allclose(result[3].poly_coef, [0, 6, 6, 2, 0, 0])
    assert np.allclose(result[4].poly_coef, [15, 10, 0, 1.66666667, 0, 0.33333333])


def test_solve_partial():
    params = GlobalParams(ppdt_enable_deg2=True)
    X = pd.DataFrame({'a': [-3, -2, -1, 0, 1, 2, 3]})
    y = np.array([0, 0, 0, 0, 1, 2, 3])
    result = solve_reg(X, y, params)
    assert len(result) == 1
    assert np.allclose(result[0].poly_coef, [0, 0, 0])
    assert result[0].fail_count == 3

    X = pd.DataFrame({'a': [1, 2, 3]})
    y = np.array([1, 2, 3])
    result = solve_reg(X, y, params)
    assert len(result) == 1
    assert np.allclose(result[0].poly_coef, [0, 1, 0])
    assert result[0].fail_count == 0


def test_inference():
    params = GlobalParams(ppdt_enable_deg2=True)
    coef = np.array([1, 2, 3, 4, 5, 6])
    poly_X = make_reg_features(X, params)
    result = np.matmul(poly_X, coef)
    assert result[0] == 64
    assert result[1] == 135
    assert result[2] == 236


def test_regex():
    params = GlobalParams(ppdt_enable_deg2=True)
    X2 = pd.DataFrame({'a': [0, 1, 2, 3], 'b': [3, 4, 5, 6]})
    y = np.array([18, 32, 50, 72])
    result = solve_reg(X2, y, params, deg2_pattern=re.compile('a|b'))
    assert len(result) == 5
    assert np.allclose(result[0].poly_coef, [0, 0, 0, 0, 0, 2])
    assert np.allclose(result[1].poly_coef, [0, 0, 6, 0, 2, 0])
    assert np.allclose(result[2].poly_coef, [18, 0, 0, -2, 4, 0])
    assert np.allclose(result[3].poly_coef, [0, 6, 6, 2, 0, 0])
    assert np.allclose(result[4].poly_coef, [15, 10, 0, 1.66666667, 0, 0.33333333])

    result = solve_reg(X2, y, params, deg2_pattern=re.compile('b'))
    assert len(result) == 3
    assert np.allclose(result[0].poly_coef, [0, 0, 0, 0, 2])
    assert np.allclose(result[1].poly_coef, [0, 0, 6, 2, 0])
    assert np.allclose(result[2].poly_coef, [18, 6, 0, 2, 0])

    result = solve_reg(X2, y, params, deg2_pattern=re.compile('c'))
    assert len(result) == 0


def test_cls():
    params = GlobalParams()
    X = pd.DataFrame({'x1': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
                      'x2': [1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 3]})
    y = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    result = solve_cls(X, np.logical_not(y), params)
    assert len(result) == 1
    assert np.allclose(result[0].poly_coef, [-1, 1, -1])
    result = solve_cls(X, y, params)
    assert len(result) == 1
    assert np.allclose(result[0].poly_coef, [0, -1, 1])
