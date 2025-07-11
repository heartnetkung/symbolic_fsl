import numpy as np
import pandas as pd
from ..util import *
from ...arc.ml.model.epdt_factory import make_epdts, LabelType
from ...arc.base import GlobalParams
import pytest

reg = LabelType.regression
cls_ = LabelType.classification


def test_step_function():
    x1 = np.arange(-10, 10)
    X = pd.DataFrame({'x1': x1})
    y = np.where(x1 < 0, 0, 5)

    regs = make_epdts(X, y, GlobalParams(epdt_max_classifer_choices=1), reg)
    assert len(regs) == 1
    assert np.allclose(y, regs[0].predict(X))
    assert regs[0].code == 'if -x1 >= 1:\n  return 0\nelse:\n  return 5'


def test_relu_function():
    x1 = np.arange(-10, 10)
    X = pd.DataFrame({'x1': x1})
    y = np.where(x1 < 0, 0, x1)

    regs = make_epdts(X, y, GlobalParams(epdt_max_classifer_choices=1), reg)
    assert len(regs) == 1
    assert np.allclose(y, regs[0].predict(X))
    assert regs[0].code == 'if -x1 >= 0:\n  return 0\nelse:\n  return x1'


def test_relu_function2():
    x1 = np.arange(-10, 10)
    X = pd.DataFrame({'x1': x1})
    y = np.where(x1 < 0, 0, x1)

    regs = make_epdts(X, y, GlobalParams(epdt_max_classifer_choices=1), cls_)
    assert len(regs) == 0
