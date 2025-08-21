from ...constant import GlobalParams
import pandas as pd
import numpy as np
from .ml_model import MLModel
from ..linprog import *


class PolynomialRegressor(MLModel):
    def __init__(self, X: pd.DataFrame, coefs: np.ndarray,
                 params: GlobalParams)->None:
        super().__init__(params)
        self.coefs = coefs
        self.columns = make_reg_columns(X, self.params)

    def _to_code(self) -> str:
        return 'return '+_poly_to_str(self.columns, self.coefs)

    def _predict(self, X: pd.DataFrame)->np.ndarray:
        return np.matmul(make_reg_features(X.fillna(0), self.params), self.coefs)

    def _get_used_columns(self)->list[str]:
        return _get_used_columns(self.columns, self.coefs)


class PolynomialClassifier(MLModel):
    def __init__(self, X: pd.DataFrame, coefs: np.ndarray, is_greater_than: bool,
                 params: GlobalParams)->None:
        super().__init__(params)
        self.coefs = coefs
        self.columns = make_cls_columns(X, self.params)
        self.is_greater_than = is_greater_than

    def _to_code(self) -> str:
        coef0 = _poly_to_str(['1'], -self.coefs[:1])
        coefs2 = self.coefs.copy()
        coefs2[0] = 0

        if self.is_greater_than:
            return f'if {_poly_to_str(self.columns, coefs2)} >= {coef0}:'
        return f'if {_poly_to_str(self.columns, coefs2)} < {coef0}:'

    def _predict(self, X: pd.DataFrame)->np.ndarray:
        result = np.matmul(make_cls_features(X.fillna(0), self.params), self.coefs)
        is_equal, is_greater = np.isclose(result, 0), result > 0
        if self.is_greater_than:
            return np.logical_or(is_greater, is_equal)
        return np.logical_not(np.logical_or(is_greater, is_equal))

    def _get_used_columns(self)->list[str]:
        return _get_used_columns(self.columns, self.coefs)


class PPDT(MLModel):
    '''Ensemble model composing of classifiers and regressors'''

    def __init__(self, classifiers: list[MLModel], regressors: list[MLModel],
                 params: GlobalParams)->None:
        super().__init__(params)
        self.cls = classifiers
        self.regs = regressors

    def predict(self, X: pd.DataFrame)->np.ndarray:
        assert len(self.cls) + 1 == len(self.regs)
        result = self.regs[-1].predict(X)
        for i in range(len(self.cls)-1, -1, -1):
            mask = self.cls[i].predict(X)
            result[mask] = self.regs[i].predict(X)[mask]
        return result

    def _to_code(self) -> str:
        assert len(self.cls) + 1 == len(self.regs)
        ans = []
        for i in range(len(self.cls)):
            elif_ = 'el' if i > 0 else ''
            ans.append('{}{}'.format(elif_, self.cls[i].code))
            ans.append('  '+self.regs[i].code)

        if len(self.cls) > 0:
            ans.append('else:')
            ans.append('  '+self.regs[-1].code)
        else:
            ans.append(self.regs[-1].code)
        return '\n'.join(ans)


def _get_used_columns(columns: list[str], coefs: np.ndarray)->list[str]:
    result = []
    for column, coef in zip(columns, coefs):
        if column == '1':
            continue
        if np.isclose(0, coef):
            continue
        result.append(column)
    return result


def _poly_to_str(columns: list[str], coefs: np.ndarray)->str:
    result = []
    for column, coef in zip(columns, coefs):
        if np.isclose(0, coef):
            continue
        if column == '1':
            result.append(_weight_to_str(coef))
        elif np.isclose(1, coef):
            result.append(column)
        elif np.isclose(-1, coef):
            result.append(f'-{column}')
        else:
            result.append(f'{_weight_to_str(coef)}*{column}')
    if len(result) == 0:
        result = ['0']
    return ' + '.join(result).replace('+ -', '- ')


def _weight_to_str(num: float)->str:
    ans = np.format_float_positional(num, 5, unique=False, trim='-')
    if ans[-1] == '.':
        return ans[:-1]
    return ans
