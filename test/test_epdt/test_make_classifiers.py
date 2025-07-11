import numpy as np
import pandas as pd
from ..util import *
from ...arc.ml.model.epdt_factory import make_classifiers
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
