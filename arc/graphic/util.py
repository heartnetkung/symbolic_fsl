from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Union
from inspect import ismethod, getmembers, isfunction
from copy import deepcopy
from .types import Grid
from ..constant import NULL_COLOR

BANNED_PROP = {'mass', 'top_color', 'shape_type', 'shape_value'}


class RuntimeObject(ABC):
    '''An interface for working with DAG'''

    @abstractmethod
    def to_entropy_var(self)->dict[str, Any]:
        '''
        List minimal variables required to represent the object.
        This is used to measure the entropy aggregated across the dataset.
        Subclasses should share common variable names among other classes.
        '''
        pass

    def to_input_var(self)->dict[str, Any]:
        '''
        List all variables generatable from this object including @property-decorated ones.
        When reason about this object, these variables will be used as the input.
        '''
        result = {}
        for name, value in getmembers(self):
            if ismethod(value) or isfunction(value) or name[0] == '_':
                continue
            result[name] = value
        return result

    def __repr__(self)->str:
        constructor_vars = [v for k, v in self.__dict__.items()
                            if k[0] != '_' and k not in BANNED_PROP]
        print(self.__dict__.items())
        params = [repr(value) for value in constructor_vars]
        return '\n{}({})'.format(self.__class__.__name__, ','.join(params))


def valid_color(c: int)->bool:
    return c < 10 and c >= 0


def make_grid(w: int, h: int, color: int = NULL_COLOR)->Grid:
    return Grid([[color for _ in range(w)] for _ in range(h)])


def geom_transform_all(grid: Grid, exclude_original: bool = False,
                       include_inverse: bool = False)->list[Grid]:
    extra_transform = []
    if include_inverse:
        inverse = grid.inverse()
        if inverse is not None:
            extra_transform = [inverse]

    if exclude_original:
        return [grid.transpose(),
                grid.flip_h(), grid.flip_h().transpose(),
                grid.flip_v(), grid.flip_v().transpose(),
                grid.flip_both(), grid.flip_both().transpose()]+extra_transform
    return [grid, grid.transpose(),
            grid.flip_h(), grid.flip_h().transpose(),
            grid.flip_v(), grid.flip_v().transpose(),
            grid.flip_both(), grid.flip_both().transpose()]+extra_transform


class Deduplicator:
    '''A simple wrapper around set for efficiently deduplicate.'''

    def __init__(self)->None:
        self.record = set()

    def has_seen_before(self, key: Any)->bool:
        len_before = len(self.record)
        self.record.add(key)
        return len_before == len(self.record)
