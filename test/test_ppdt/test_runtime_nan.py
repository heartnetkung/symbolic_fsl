from ...arc.ml import *
from ...arc.ml.model.association_factory import *
from ...arc.ml.model.comparison_factory import *
from ...arc.ml.model.base_models import *
from ...arc.ml.model.decision_tree_factory import *
from ...arc.base import *
from ...arc.constant import *
import pandas as pd
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

params = GlobalParams()
normal_df = pd.DataFrame({'a': [0, 1, 1, 0], 'b': [0, 1, 2, 3], 'c': [1, 1, 1, 1]})
label = np.array([0, 1, 1, 0])
success_df = pd.DataFrame({'a': [0, 1, 1, 0], 'b': [None, 1, 2, 3], 'c': [1, 1, 1, 1]})
fail_df = pd.DataFrame({'a': [None, 1, 1, 0], 'b': [0, 1, 2, 3], 'c': [1, 1, 1, 1]})


def do_test(model):
    '''
    Dataframe might contains None, in such case the model should
        1. throw error is the column is in use
        2. works normally if it's not
    '''
    assert np.array_equal(model.predict(normal_df), label)
    assert np.array_equal(model.predict(success_df), label)
    with pytest.raises(IgnoredException):
        model.predict(fail_df)


def test_column_model():
    model = ColumnModel('a')
    do_test(model)


def test_constant_column_model():
    model = ConstantColumnModel('a', 1)
    do_test(model)


def test_association():
    model = Association('a', {1: 1, 0: 0}, params)
    do_test(model)


def test_tree():
    inner_model = DecisionTreeClassifier()
    inner_model.fit(normal_df, label)
    model = DTClassifier(inner_model, normal_df.columns)
    do_test(model)


def test_comparison():
    model = ComparisonModel('a', 'c', True, params)
    do_test(model)
    model2 = ConstantComparisonModel('a', 1, True, params)
    do_test(model2)


def test_reg():
    results = solve_reg(normal_df, label, params)
    model = PolynomialRegressor(normal_df, results[0].poly_coef, params)
    do_test(model)


def test_cls():
    results = solve_cls(normal_df, label, params)
    model = PolynomialClassifier(normal_df, results[0].poly_coef, True, params)
    do_test(model)
