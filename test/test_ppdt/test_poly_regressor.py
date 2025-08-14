import pandas as pd
import numpy as np
from ...arc.ml.model.ppdt_factory import make_regressors, LabelType
from ..util import *
import pytest
from ...arc.base import GlobalParams
import re

reg = LabelType.reg
cls_ = LabelType.cls_


def test_constant():
    len_ = 5
    X = pd.DataFrame({'x1': np.arange(len_)})
    y = np.full(len_, 3)

    regressors = make_regressors(X, y, GlobalParams(), reg)
    assert 'return 3' == regressors[0].code
    assert np.allclose(y, regressors[0].predict(X))
    assert len(regressors) == 1


def test_noisy_constant():
    len_ = 5
    X = pd.DataFrame({'x1': np.arange(len_),
                      'x2': random_with_seed(len_, seed=0),
                      'x3': random_with_seed(len_, seed=1)})
    y = np.full(len_, 3)
    regressors = make_regressors(X, y, GlobalParams(), reg)
    assert 'return 3' == regressors[0].code
    assert np.allclose(y, regressors[0].predict(X))
    assert len(regressors) == 1


def test_sum_interpolation():
    len_ = 5
    X = pd.DataFrame({'x1': np.arange(len_), 'x2': random_with_seed(len_)})
    y = np.array(3*X['x1'] + 2*X['x2'])
    regressors = make_regressors(X, y, GlobalParams(), reg)
    assert len(regressors) == 1
    assert 'return 3*x1 + 2*x2' == regressors[0].code
    assert np.allclose(y, regressors[0].predict(X))


def test_sum_interpolation2():
    len_ = 5
    X = pd.DataFrame({'x1': np.arange(len_), 'x2': random_with_seed(len_)})
    y = np.array(3*X['x1'] + 2*X['x2'])
    regressors = make_regressors(X, y, GlobalParams(), cls_)
    assert len(regressors) == 1
    assert re.match(r'return [\d.]+', regressors[0].code)


def test_noisy_sum_interpolation():
    len_ = 10
    X = pd.DataFrame({'x1': np.arange(len_),
                      'x2': random_with_seed(len_, seed=0),
                      'x3': random_with_seed(len_, seed=1),
                      'x4': random_with_seed(len_, seed=2)})
    y = np.array(3*X['x1'] + 2*X['x2'])
    regressors = make_regressors(X, y, GlobalParams(), reg)
    assert len(regressors) == 2
    assert re.match(r'return [\d.]+', regressors[0].code)
    assert 'return 3*x1 + 2*x2' == regressors[1].code
    assert np.allclose(y, regressors[1].predict(X))


def test_noisy_sum_interpolation():
    len_ = 10
    X = pd.DataFrame({'x1': np.arange(len_),
                      'x2': random_with_seed(len_, seed=0),
                      'x3': random_with_seed(len_, seed=1),
                      'x4': random_with_seed(len_, seed=2)})
    y = np.array(3*X['x1'] + 2*X['x2'])
    regressors = make_regressors(X, y, GlobalParams(), cls_)
    assert len(regressors) == 1
    assert re.match(r'return [\d.]+', regressors[0].code)
