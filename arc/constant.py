from __future__ import annotations
from functools import cached_property
from numpy.random import RandomState, SeedSequence, MT19937
import numpy as np
import pandas as pd
import re
from enum import Enum
from typing import Any, Iterable
from dataclasses import dataclass, fields, asdict, replace

ANY_PATTERN = re.compile('.*')
MISSING_VALUE = -5
NULL_COLOR = -1
NULL_DF = pd.DataFrame([])
BOOLS = [False, True]
MAX_SHAPES_PER_GRID = 30
MAX_REPARSE_EDGE = 300
COST_PATTERN = re.compile(r'\*|\+|\- | < | > |==|<=|>=|!=')


class ParseMode(Enum):
    proximity_diag = 0
    proximity_normal = 1
    color_proximity_diag = 2
    color_proximity_normal = 3
    crop = 4
    partition = 5


@dataclass(frozen=True)
class GlobalParams:
    '''
    Store all modifiable parameters of all classes for hyperparameter tuning.
    '''

    # random seed
    seed: int = 0
    # enable epdt to try polynomial of degree 2
    ppdt_enable_deg2: bool = False
    # the number of possible classifiers per branch
    ppdt_max_classifer_choices: int = 2
    # the number of possible regressors per branch
    ppdt_max_regressor_choices: int = 1
    # the number of possible regressors per PPDT
    ppdt_max_nested_regressors: int = 3
    # the depth of decision tree used in PPDT
    ppdt_decision_tree_depth: int = 2
    # linear programming time limit
    linprog_time_limit: int = 10
    # enable_free_draw
    enable_free_draw: bool = True
    # maximum reparse operations per solution
    # max_reparse: int = 0
    max_reparse: int = 2
    # list of parse modes to try
    # parser_x_modes: Iterable[ParseMode] = (ParseMode.crop,)
    # parser_y_modes: Iterable[ParseMode] = (ParseMode.crop,)
    parser_x_modes: Iterable[ParseMode] = ParseMode
    parser_y_modes: Iterable[ParseMode] = ParseMode

    @cached_property
    def nprandom(self):
        return RandomState(MT19937(SeedSequence(self.seed)))

    def random(self, size: int)->np.ndarray:
        '''Random a numpy array with given size with value from 0-1.'''
        return self.nprandom.rand(size)

    def shuffle(self, arr: np.ndarray)->None:
        '''Shuffle the given array.'''
        self.nprandom.shuffle(arr)

    def __repr__(self)->str:
        return 'GlobalParams()'

    def update(self, **kwargs)->GlobalParams:
        return replace(self, **kwargs)


class FuzzyBool(Enum):
    no = 0
    yes = 1
    maybe = 2


def default_repr(obj: Any)->str:
    vars_ = [f'{k}={v}' for k, v in obj.__dict__.items() if k[0] != '_']
    return '{}({})'.format(obj.__class__.__name__, ','.join(vars_))


def default_hash(obj: Any)->int:
    comparable_fields = [f.name for f in fields(obj) if f.compare]
    dict_ = asdict(obj)
    values = tuple(repr(dict_[f]) for f in comparable_fields)
    return hash(values)


class IgnoredException(Exception):
    pass
