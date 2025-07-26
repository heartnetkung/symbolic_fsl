from abc import ABC, abstractmethod
from ...constant import GlobalParams
from functools import cached_property
import numpy as np
import pandas as pd
from ...graphic import Grid


class VisualRepresentation(ABC):
    '''
    Represents a method of data representation.
    '''

    @abstractmethod
    def encode_feature(self, grids: list[Grid], feature: pd.DataFrame)->pd.DataFrame:
        pass

    @abstractmethod
    def encode_label(self, grids: list[Grid], label: np.ndarray)->np.ndarray:
        pass

    @abstractmethod
    def decode_label(self, grids: list[Grid], label: np.ndarray)->np.ndarray:
        pass


class VisualModel(ABC):
    '''
    Represents a set of machine learning models that predict visual data.
    For visual data, the input and output can be represented in multiple ways.
    For example, to find a coordinate (x,y) we can either predict the number itself,
    or scan the image to find a specific pattern around the target.

    Thus, having this abstraction helps encapsulate both the machine learning part
    ans the representational encoding.
    '''

    def __init__(self, rep: VisualRepresentation)->None:
        self.rep = rep

    @cached_property
    def code(self) -> str:
        return self._to_code()

    def __repr__(self)->str:
        return self.code

    def predict(self, grids: list[Grid], X: pd.DataFrame)->np.ndarray:
        X2 = self.rep.encode_feature(grids, X)
        y = self._raw_predict(grids, X2)
        return self.rep.decode_label(grids, y)

    @abstractmethod
    def _to_code(self) -> str:
        '''
        Transform the current fitted state to Python code.
        '''
        pass

    @abstractmethod
    def _raw_predict(self, grids: list[Grid], X: pd.DataFrame)->np.ndarray:
        '''
        Use the current fitted state to predict new output.
        Note: len(X) must be equal to len(grids).
        Note2: np.ndarray is a 2 dimensional array.
        '''
        pass


class MemorizedVModel(VisualModel):
    '''Model that memorized answer.'''

    def __init__(self, result: np.ndarray)->None:
        self.result = result

    def _to_code(self) -> str:
        return f'return {self.result}'

    def _raw_predict(self, grids: list[Grid], X: pd.DataFrame)->np.ndarray:
        return np.array([])

    def predict(self, grids: list[Grid], X: pd.DataFrame)->np.ndarray:
        return self.result


class DummyRepresentation(VisualRepresentation):
    def encode_feature(self, grids: list[Grid], feature: pd.DataFrame)->pd.DataFrame:
        return feature

    def encode_label(self, grids: list[Grid], label: np.ndarray)->np.ndarray:
        return label

    def decode_label(self, grids: list[Grid], label: np.ndarray)->np.ndarray:
        return label
