from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from ...constant import GlobalParams, MISSING_VALUE
from typing import Any, Optional, Callable, TypeVar
import inspect
from functools import cached_property


CLS_FIELD_SUFFIX = '_cls'
T = TypeVar('T')


class MLModel(ABC):
    '''
    Represents a machine learning model that is expressible as Python code.
    It can fit its parameter to the given dataset, predict new output,
    as well as turn its logic into code.
    '''

    def __init__(self, params: GlobalParams)->None:
        self.params = params

    @abstractmethod
    def _to_code(self) -> str:
        '''
        Transform the current fitted state to Python code.
        '''
        pass

    @cached_property
    def code(self) -> str:
        return self._to_code()

    def __repr__(self)->str:
        return self.code

    @abstractmethod
    def predict(self, X: pd.DataFrame)->np.ndarray:
        '''
        Use the current fitted state to predict new output.
        '''
        pass

    def predict_int(self, X: pd.DataFrame)->list[int]:
        '''Predict int values'''
        return [int(round(result)) for result in self.predict(X)]

    def predict_enum(self, X: pd.DataFrame, enum_type: type[T])->list[T]:
        '''Predict type-safe enum values'''
        try:
            return [enum_type(result) for result in self.predict_int(X)]  # type:ignore
        except ValueError:
            print('Enum value out of bound. Only class enum is supported.')
            raise

    def predict_bool(self, X: pd.DataFrame)->list[bool]:
        '''Predict boolean values'''
        return [bool(round(result)) for result in self.predict(X)]


class ConstantModel(MLModel):
    '''Model that always returns a constant value regardless of input.'''

    def __init__(self, output: Any)->None:
        self.output = output

    def _to_code(self) -> str:
        return f'return {self.output}'

    def predict(self, X: pd.DataFrame)->np.ndarray:
        return np.full(X.shape[0], self.output)


class ColumnModel(MLModel):
    '''Model that select a column from the input and return.'''

    def __init__(self, col_name: str)->None:
        self.col_name = col_name

    def _to_code(self) -> str:
        return f'return {self.col_name}'

    def predict(self, X: pd.DataFrame)->np.ndarray:
        col = X[self.col_name]
        assert col is not None
        return col.to_numpy()


class MemorizedModel(MLModel):
    '''Model that memorized answer.'''

    def __init__(self, result: np.ndarray)->None:
        self.result = result

    def _to_code(self) -> str:
        return f'return {self.result}'

    def predict(self, X: pd.DataFrame)->np.ndarray:
        if len(X) != len(self.result):
            raise Exception('length mismatched')
        return self.result


class FunctionModel(MLModel):
    '''
    Model represented by a deterministic function used for testing.
    The return type can be list, pd.Series, or np.ndarray
    '''

    def __init__(self, predictor: Callable,
                 ensure_columns: Optional[list[str]] = None)->None:
        self.predictor = predictor
        self.ensure_columns = ensure_columns

    def _to_code(self) -> str:
        src = inspect.getsource(self.predictor)
        if 'lambda' in src:
            src = 'lambda' + src.split('lambda')[-1].rstrip()
        return f'FunctionModel({src})'

    def predict(self, X: pd.DataFrame)->np.ndarray:
        if self.ensure_columns is not None:
            existing_columns = set(X.columns)
            required_columns = set(self.ensure_columns)
            extra_columns = required_columns-existing_columns
            appending_data = {col: [MISSING_VALUE]*X.shape[0] for col in extra_columns}
            X = pd.concat((X.reset_index(drop=True),
                           pd.DataFrame(appending_data)), axis=1)
        result = self.predictor(X)
        if isinstance(result, pd.Series):
            return result.to_numpy()
        elif isinstance(result, np.ndarray):
            return result
        elif isinstance(result, list):
            return np.array(result)
        raise Exception('unsupported return type')
