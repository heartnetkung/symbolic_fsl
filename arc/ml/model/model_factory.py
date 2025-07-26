from .ml_model import *
from .ppdt_factory import *
from itertools import combinations
from .ml_model import MLModel, ConstantModel
from typing import Optional, Callable
from ...constant import GlobalParams, MISSING_VALUE
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import lru_cache
import logging
from .association_factory import make_association
from .decision_tree_factory import make_tree

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    X: pd.DataFrame
    y: np.ndarray
    params: GlobalParams

    def __eq__(self, other):
        if other.__class__ != TrainingData:
            return False
        return self.X.equals(other.X) and np.array_equal(self.y, other.y)

    def __hash__(self):
        return hash(f'{self.X} {self.y}')


def classifier_factory(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
                       print_str: str)->list[MLModel]:
    if y.dtype.name == 'bool':
        y = y.astype(int)
    return _model_factory(X, y, params, print_str, LabelType.classification)


def regressor_factory(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
                      print_str: str)->list[MLModel]:
    return _model_factory(X, y, params, print_str, LabelType.regression)


def _model_factory(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
                   print_str: str, type: LabelType)->list[MLModel]:
    if len(X) != len(y):
        return []

    # zero column answer
    y_set = set(y)
    if len(y_set) == 1:
        return [ConstantModel(y_set.pop())]

    # exact answer
    exact_result = []
    for col in X.columns:
        if np.allclose(y, X[col]):
            exact_result.append(ColumnModel(col))
    if len(exact_result) > 0:
        return exact_result

    # preprocess
    X = _drop_constants(X)
    X2 = _drop_repeated(X)
    if X2.empty:
        return []

    # TODO drop duplicates

    logger.info('solving: %s %s %s', X2.shape, y, print_str)

    ppdt_models = make_models(TrainingData(X2, y, params), type)
    assoc_models = make_association(X, y, params)
    tree_models = [MatchColumn(FeatEng(model), X2)
                   for model in make_tree(feat_eng(X2), y, params)]

    logger.info('ppdt_models: %s', ppdt_models)
    logger.info('assoc_models: %s', assoc_models)
    logger.info('tree_models: %s', tree_models)
    return ppdt_models+assoc_models+tree_models


@lru_cache
def make_models(data: TrainingData, type: LabelType)->list[MLModel]:
    result, X, y = [], data.X, data.y
    for ppdt in make_ppdts(X, y, data.params, type):
        if np.allclose(ppdt.predict(X), y):
            result.append(MatchColumn(ppdt, X))
    return result


def model_selection(*models: list[MLModel])->list[tuple[MLModel, ...]]:
    lengths = [len(x) for x in models]
    if min(lengths) == 0:
        return []

    result = []
    for i in range(max(lengths)):
        unit = []
        for variations in models:
            unit.append(variations[i % len(variations)])
        result.append(tuple(unit))
    return result

# ============================
# Column manipulation
# ============================


class FeatEng(MLModel):
    def __init__(self, inner_model: MLModel)->None:
        self.inner_model = inner_model

    def predict(self, X: pd.DataFrame)->np.ndarray:
        return self.inner_model.predict(feat_eng(X))

    def _to_code(self) -> str:
        return self.inner_model.code


class MatchColumn(MLModel):
    def __init__(self, inner_model: MLModel, X: pd.DataFrame)->None:
        self.inner_model = inner_model
        self.trained_columns = list(X.columns)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        X = _match_column(X, self.trained_columns)
        return self.inner_model.predict(X)

    def _to_code(self) -> str:
        return self.inner_model.code


def _match_column(X: pd.DataFrame, expected_columns: list[str])->pd.DataFrame:
    existing_cols = set(X.columns)
    expected_cols = set(expected_columns)
    if existing_cols == expected_cols:
        return X

    dropping_cols = existing_cols-expected_cols
    if len(dropping_cols) > 0:
        X = X.drop(list(dropping_cols), axis=1)

    appending_cols = expected_cols-existing_cols
    if len(appending_cols) > 0:
        appending = {col: [MISSING_VALUE]*X.shape[0] for col in appending_cols}
        X = pd.concat((X, pd.DataFrame(appending)), axis=1)

    return X[expected_columns].copy()  # type:ignore


def _drop_redundants(df: pd.DataFrame)->pd.DataFrame:
    to_drop = set()
    for col in df.columns:
        if len(set(df[col])) == 1:
            to_drop.add(col)

    leftover_cols = [col for col in df.columns if col not in to_drop]
    for col1, col2 in combinations(leftover_cols, 2):
        if np.array_equal(df[col1], df[col2]):
            to_drop.add(col2)
    return df.drop(list(to_drop), axis=1)


def _drop_constants(df: pd.DataFrame)->pd.DataFrame:
    to_drop = set()
    for col in df.columns:
        if len(set(df[col])) == 1:
            to_drop.add(col)
    return df.drop(list(to_drop), axis=1)


def _drop_repeated(df: pd.DataFrame)->pd.DataFrame:
    to_drop = set()
    for col1, col2 in combinations(df.columns, 2):
        if np.array_equal(df[col1], df[col2]):
            to_drop.add(col2)
    return df.drop(list(to_drop), axis=1)


def feat_eng(df: pd.DataFrame)->pd.DataFrame:
    result = {}
    for col1, col2 in combinations(df.columns, 2):
        result[f'({col1} == {col2})'] = np.where(df[col1] == df[col2], 1, 0)
    return pd.concat([df, pd.DataFrame(result)], axis=1)
