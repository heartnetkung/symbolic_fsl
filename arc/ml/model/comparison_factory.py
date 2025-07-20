import numpy as np
import pandas as pd
from .ml_model import MLModel
from itertools import combinations
from ...constant import GlobalParams


class ComparisonModel(MLModel):
    '''
    A simple work around since linear classifier cannot solve xor problem.
    It is simply exhaustively check every combination of fields.
    '''

    def __init__(self, col1: str, col2: str, eq: bool, params: GlobalParams)->None:
        super().__init__(params)
        self.col1 = col1
        self.col2 = col2
        self.eq = eq

    def predict(self, X: pd.DataFrame)->np.ndarray:
        if self.eq:
            return X[self.col1] == X[self.col2]
        return X[self.col1] != X[self.col2]

    def _to_code(self) -> str:
        if self.eq:
            return f'if {self.col1} == {self.col2}:'
        return f'if {self.col1} != {self.col2}:'


def make_comparison_models(X: pd.DataFrame, y: np.ndarray,
                           params: GlobalParams)->list[MLModel]:
    result = []
    for col1, col2 in combinations(X.columns, 2):
        for eq in [True, False]:
            candidate = ComparisonModel(col1, col2, eq, params)
            if np.array_equal(candidate.predict(X), y):
                result.append(candidate)
    return result
