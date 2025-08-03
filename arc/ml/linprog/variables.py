import pandas as pd
import numpy as np
from scipy.optimize import Bounds
from dataclasses import dataclass
from ...constant import COST_PATTERN

C_MAX = 10  # ref III.1.2.c
C0_MAX = 30


@dataclass
class Variables:
    '''Bundle class storing all optimizing variables'''
    cost: list[float]
    integrality: list[int]
    bounds: Bounds
    columns: list[str]

    def __repr__(self)->str:
        data = {'cost': self.cost, 'integ': self.integrality,
                'lb': self.bounds.lb, 'ub': self.bounds.ub}
        return f'{Variables}\n{pd.DataFrame(data, index=self.columns)}'  # type:ignore

    def result2str(self, result: np.ndarray)->str:
        data = {'result': result}
        return f'{Variables}\n{pd.DataFrame(data, index=self.columns)}'  # type:ignore


class VariableCount:
    def __init__(self, X: pd.DataFrame, deg2_cols: list[tuple[str, str]],
                 is_cls: bool = False)->None:
        self.n_col = X.shape[1]
        self.n_sample = X.shape[0]
        # coef associated with polynomial degree 0,1,2
        self.c0 = 2
        self.c1 = 2*self.n_col
        self.c2 = 2*len(deg2_cols)
        # constrainst satisfaction boolean variables
        self.b = 0 if is_cls else self.n_sample
        self.t = 0 if is_cls else 1
        # total count
        self.c_total = self.c0+self.c1+self.c2
        self.total = self.c0+self.c1+self.c2+self.b+self.t
        # coef associated with polynomial degree 0,1,2 on the plus side
        self.c0p = int(round(self.c0/2))
        self.c1p = int(round(self.c1/2))
        self.c2p = int(round(self.c2/2))

    def __repr__(self)->str:
        return repr(self.__dict__)


def make_variables(counts: VariableCount, is_cls: bool,
                   col_names: list[str], lambda_: int = 0)->Variables:
    # configure
    # should the cost be zero? No because 224.
    c0_cost = 0 if is_cls else 1
    c_integrality = 1 if is_cls else 0
    c_max = 1 if is_cls else C_MAX

    c1_cost = [2+_cal_penalty(col_name) for col_name in col_names]*2
    # TODO relax integrality constraints?

    # make variables
    lb = [0]*counts.total
    ub = [C0_MAX]*counts.c0 + [c_max]*(counts.c1+counts.c2) + [1]*(counts.b+counts.t)
    integrality = [c_integrality]*counts.c_total + [1]*(counts.b+counts.t)
    cost = ([c0_cost]*counts.c0 +
            c1_cost +
            [3]*counts.c2 +
            [lambda_]*(counts.b+counts.t))

    # make columns
    columns = ['c0p', 'c0m']
    columns += [f'c1{i}p' for i in range(counts.c1p)]
    columns += [f'c1{i}m' for i in range(counts.c1p)]
    columns += [f'c2{i}p' for i in range(counts.c2p)]
    columns += [f'c2{i}m' for i in range(counts.c2p)]
    columns += [f'b{i}' for i in range(counts.b)]
    columns += [f't{i}' for i in range(counts.t)]

    return Variables(cost, integrality, Bounds(lb, ub), columns)  # type:ignore


def _cal_penalty(col_name: str)->int:
    return len(COST_PATTERN.split(col_name))-1
