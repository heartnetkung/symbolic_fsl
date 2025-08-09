import numpy as np
import pandas as pd
from ...constant import GlobalParams
from .ml_model import MLModel, CLS_FIELD_SUFFIX, ConstantModel, ColumnModel
from ..linprog import *
from .base_models import *
from scipy.stats import mode
import re
from enum import Enum
from itertools import product
from typing import Optional
from .comparison_factory import make_comparison_models
from .decision_tree_factory import make_tree


DECIMAL_FILTER = re.compile(r'\d\.\d\d+')
MAX_LINPROG_REG_SAMPLE = 100  # linprog reg is not really scalable and mostly timeout
MAX_CLASSIFIERS = 5


class LabelType(Enum):
    '''
    LabelType specify the type of label useful for contraining the training process.
    In regression, addition and multiplication are allowed in "then" clause.
    In classification, they are not allowed because the numerical value does not matter.
    '''
    classification = 0
    regression = 1


def make_ppdts(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
               type: LabelType)->list[MLModel]:
    '''List all possible EDPT models that perfectly fit the data.'''
    result = []
    reg_columns = [col for col in X.columns if not col.endswith(CLS_FIELD_SUFFIX)]
    X_reg = X[reg_columns].copy()
    assert isinstance(X_reg, pd.DataFrame)

    all_regressors = []
    _make_all_regressors(X_reg, y, params, type, [], all_regressors)
    for regressors in all_regressors:
        all_classifiers = _make_all_classifiers(X, y, params, regressors)
        for classifier_comb in product(*all_classifiers):
            result.append(PPDT(list(classifier_comb), regressors, params))
    return result


def _make_all_regressors(
        X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
        type: LabelType, prefix: list[MLModel], result: list[list[MLModel]])->None:
    if len(prefix) == params.ppdt_max_nested_regressors:
        return

    for regressor in make_regressors(X, y, params, type):
        correct_pred = np.isclose(y, regressor.predict(X))
        if np.all(correct_pred):
            result.append(prefix+[regressor])
            break
        if not np.any(correct_pred):
            continue

        wrong_pred = np.logical_not(correct_pred)
        X_remain = X[wrong_pred].reset_index(drop=True)
        assert isinstance(X_remain, pd.DataFrame)
        y_remain = y[wrong_pred].copy()
        next_prefix = prefix+[regressor]
        _make_all_regressors(X_remain, y_remain, params, type, next_prefix, result)


def make_regressors(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
                    type: LabelType)->list[MLModel]:
    if len(y) == 0:
        return []

    max_result = params.ppdt_max_regressor_choices
    result: list[MLModel] = []

    if type == LabelType.regression and (len(y) in range(2, MAX_LINPROG_REG_SAMPLE)):
        for raw_result in solve_reg(X, y, params, max_result=max_result):
            new_model = PolynomialRegressor(X, raw_result.poly_coef, params)
            if DECIMAL_FILTER.search(new_model.code) is None:
                result.append(new_model)

    if len(result) == 0:
        return _select_trivial_values(X, y, type)
    return result


def _select_trivial_values(X: pd.DataFrame, y: np.ndarray,
                           type: LabelType)->list[MLModel]:
    mode_result, (n_row, n_col) = mode(y), X.shape
    match_count = mode_result.count+1  # mode has 1 extra score since it's simpler
    result: list[MLModel] = [ConstantModel(mode(y).mode)]
    if (match_count >= n_row) or (type == LabelType.classification):
        return result

    single_column_model: Optional[ColumnModel] = None
    for col in X.columns:
        new_count = np.sum(X[col] == y)
        if new_count > match_count:
            single_column_model = ColumnModel(col)
            match_count = new_count

    if single_column_model is not None:
        result.append(single_column_model)

    return result


def _make_all_classifiers(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
                          regressors: list[MLModel])->list[list[MLModel]]:
    assert len(regressors) > 0
    result, X_remain, y_remain = [], X, y
    depth = params.ppdt_decision_tree_depth

    for regressor in regressors[:-1]:
        assert isinstance(X_remain, pd.DataFrame)
        correct_pred = np.isclose(y_remain, regressor.predict(X_remain))
        classifiers1 = make_classifiers(X_remain, correct_pred, params)
        classifiers2 = make_tree(X_remain, correct_pred, params, depth)
        classifiers3 = make_comparison_models(X_remain, correct_pred, params)
        all_classifiers = classifiers1+classifiers2+classifiers3
        result.append(all_classifiers[:MAX_CLASSIFIERS])

        wrong_pred = np.logical_not(correct_pred)
        X_remain = X_remain[wrong_pred].reset_index(drop=True)
        y_remain = y_remain[wrong_pred].copy()
    return result


def make_classifiers(X: pd.DataFrame, y: np.ndarray,
                     params: GlobalParams)->list[MLModel]:
    assert not y.all(), 'classifier with all true will cause training errors'
    assert y.any(), 'classifier with all false will cause training errors'

    max_result = params.ppdt_max_classifer_choices
    raw_results = solve_cls(X, y, params, max_result=max_result)
    if len(raw_results) == 0:
        return []

    raw_results.sort(key=lambda result: result.score)
    min_score = raw_results[0].score
    candidates = filter(lambda result: result.score == min_score, raw_results)
    return [PolynomialClassifier(X, raw_result.poly_coef, True, params)
            for raw_result in candidates]
