import numpy as np
import pandas as pd
from ..util import *
from ...arc.ml.model.ppdt_factory import make_classifiers
from ...arc.ml.model.comparison_factory import make_comparison_models
from ...arc.base import GlobalParams
import pytest


def test_error_case():
    len_ = 10
    X = pd.DataFrame({'x1': random_with_seed(len_), 'x2': random_with_seed(len_)})
    with pytest.raises(Exception) as e_info:
        y = np.array([1]*len_)
        cls_ = make_classifiers(X, y, GlobalParams())
    with pytest.raises(Exception) as e_info:
        y = np.array([0]*len_)
        cls_ = make_classifiers(X, y, GlobalParams())


def test_classifier():
    x1 = np.array(range(40))
    x2 = np.where(x1 % 3 == 0, x1, np.array(range(0, 80, 2)))
    X = pd.DataFrame({'x1': x1, 'x2': x2})
    y = np.where(x1 == x2, True, False)
    classifiers = make_classifiers(X, y, GlobalParams())
    assert len(classifiers) == 1
    assert classifiers[0].code == 'if x1 - x2 >= 0:'
    assert np.allclose(classifiers[0].predict(X), y)

    x1 = np.array(range(20))
    X = pd.DataFrame({'x1': x1})
    y = np.where(x1 < 8, True, False)
    classifiers = make_classifiers(X, y, GlobalParams())
    assert len(classifiers) == 1
    assert classifiers[0].code == 'if -x1 >= -7:'
    assert np.allclose(classifiers[0].predict(X), y)


def test_xor():
    x1 = np.array([8, 2, 8, 8, 2, 2, 2, 3, 4, 1, 1, 1])
    x2 = np.array([8, 8, 8, 8, 2, 2, 2, 2, 1, 1, 1, 1])
    X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x1*x2})
    y = np.array([0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    params = GlobalParams()

    classifiers = make_comparison_models(X, y, GlobalParams())
    assert len(classifiers) == 1
    assert classifiers[0].code == 'if x1 != x2:'
    assert np.allclose(classifiers[0].predict(X), y)

    classifiers = make_comparison_models(X, np.logical_not(y), GlobalParams())
    assert len(classifiers) == 1
    assert classifiers[0].code == 'if x1 == x2:'
    assert np.allclose(classifiers[0].predict(X), np.logical_not(y))
