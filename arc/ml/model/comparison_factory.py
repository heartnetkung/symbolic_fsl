import numpy as np
import pandas as pd
from .ml_model import MLModel
from itertools import combinations
from ...constant import GlobalParams, BOOLS


class ComparisonModel(MLModel):
    '''
    A simple work around since linear classifier cannot compare
    exact equality between two fields.
    '''

    def __init__(self, col1: str, col2: str, eq: bool, params: GlobalParams)->None:
        super().__init__(params)
        self.col1 = col1
        self.col2 = col2
        self.eq = eq

    def predict(self, X: pd.DataFrame)->np.ndarray:
        if self.eq:
            return np.isclose(X[self.col1], X[self.col2])
        return np.logical_not(np.isclose(X[self.col1], X[self.col2]))

    def _to_code(self) -> str:
        if self.eq:
            return f'if {self.col1} == {self.col2}:'
        return f'if {self.col1} != {self.col2}:'


class ConstantComparisonModel(MLModel):
    '''
    A simple work around since linear classifier cannot compare
    exact equality with constant.
    '''

    def __init__(self, col: str, value: int, eq: bool, params: GlobalParams)->None:
        super().__init__(params)
        self.col = col
        self.value = value
        self.eq = eq

    def predict(self, X: pd.DataFrame)->np.ndarray:
        if self.eq:
            return np.isclose(X[self.col], self.value)
        return np.logical_not(np.isclose(X[self.col], self.value))

    def _to_code(self) -> str:
        if self.eq:
            return f'if {self.col} == {self.value}:'
        return f'if {self.col} != {self.value}:'


def make_comparison_models(X: pd.DataFrame, y: np.ndarray,
                           params: GlobalParams)->list[MLModel]:
    result = []
    for col1, col2 in combinations(X.columns, 2):
        for eq in BOOLS:
            candidate = ComparisonModel(col1, col2, eq, params)
            if np.array_equal(candidate.predict(X), y):
                result.append(candidate)

    for col in X.columns:
        all_correct_values = set(X[col][y])
        if len(all_correct_values) == 1:
            value = all_correct_values.pop()
            candidate = ConstantComparisonModel(col, value, True, params)
            if np.array_equal(candidate.predict(X), y):
                result.append(candidate)

        all_incorrect_values = set(X[col][np.logical_not(y)])
        if len(all_incorrect_values) == 1:
            value = all_incorrect_values.pop()
            candidate = ConstantComparisonModel(col, value, False, params)
            if np.array_equal(candidate.predict(X), y):
                result.append(candidate)

    return result
