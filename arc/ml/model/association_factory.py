from ...constant import GlobalParams, MISSING_VALUE, IgnoredException
import numpy as np
import pandas as pd
from .ml_model import MLModel


class Association(MLModel):
    '''A simple association model implemented using dictionary.'''

    def __init__(self, column: str, kv: dict, params: GlobalParams)->None:
        super().__init__(params)
        self.column = column
        self.kv = kv

    def predict(self, X: pd.DataFrame)->np.ndarray:
        col = X[self.column]
        if col is None:
            raise IgnoredException()

        result = []
        for k in col:
            v = self.kv.get(k)
            if v is None:
                raise IgnoredException()
            result.append(v)
        return np.array(result)

    def _to_code(self) -> str:
        ans, count = [], 0
        items = self.kv.items()
        for k, v in items:
            if count == 0:
                ans.append(f'if {self.column} == {k}:')
            elif count == len(items)-1:
                ans.append('else:')
            else:
                ans.append(f'elif {self.column} == {k}:')
            ans.append(f'  return {v}')
            count += 1
        return '\n'.join(ans)


def make_association(X: pd.DataFrame, y: np.ndarray,
                     params: GlobalParams)->list[MLModel]:
    if len(y)/len(set(y)) <= 1.4:
        return []

    result = []
    for col_name in X.columns:
        success, kv = True, {}
        for k, v in zip(X[col_name], y):
            existing_value = kv.get(k)
            if existing_value is None:
                kv[k] = v
            elif existing_value != v:
                success = False
                break

        if success:
            result.append(Association(col_name, kv, params))
    return result
