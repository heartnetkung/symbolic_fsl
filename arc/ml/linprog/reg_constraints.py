import pandas as pd
import numpy as np
from scipy.optimize import LinearConstraint
from .variables import *

POS_ENCODING = 1.3
BIG_M = 1e6
EPSILON = 1e-4
MIN_MATCH = 2


def make_dataset_constraints(
        counts: VariableCount, X: pd.DataFrame, y: np.ndarray,
        deg2_cols: list[tuple[str, str]])->list[LinearConstraint]:
    '''Match as many data points as possible.'''
    poly_feats = _make_internal_polynomial_features(counts, X, True, deg2_cols)
    poly_feats2 = _make_internal_polynomial_features(counts, X, False, deg2_cols)
    return [LinearConstraint(poly_feats, ub=y),  # type:ignore
            LinearConstraint(poly_feats2, lb=y)]  # type:ignore


def make_min_match_constraints(counts: VariableCount)->LinearConstraint:
    '''Match at least n data points.'''
    constraints = np.zeros(counts.total)
    constraints[counts.c_total:counts.c_total+counts.b] = 1
    return LinearConstraint(constraints, ub=counts.n_sample-MIN_MATCH)


def make_mistake_notone_constraints(counts: VariableCount)->list[LinearConstraint]:
    '''n_mistakes must not be one (or the next iteration is not solvable)'''
    constraints = np.zeros(counts.total)
    c, b, t = counts.c_total, counts.b, counts.t
    constraints[c:c+b] = 1
    constraints[c+b:c+b+t] = -BIG_M
    return [LinearConstraint([constraints], ub=1-EPSILON),
            LinearConstraint([constraints], lb=1+EPSILON-BIG_M)]


def _make_internal_polynomial_features(
        counts: VariableCount, X: pd.DataFrame, first_call: bool,
        deg2_cols: list[tuple[str, str]])->np.ndarray:
    result, columns = [], list(X.columns)

    # c0p, c0m
    result.append([1]*counts.n_sample)
    result.append([-1]*counts.n_sample)
    # c1p, c1m
    for col in columns:
        result.append(X[col])
    for col in columns:
        result.append(-X[col])
    # c2p, c2m
    for col1, col2 in deg2_cols:
        result.append(X[col1]*X[col2])
    for col1, col2 in deg2_cols:
        result.append(-X[col1]*X[col2])
    # b
    b_const = -BIG_M if first_call else BIG_M
    for i in range(counts.b):
        new_value = np.zeros(counts.n_sample)
        new_value[i] = b_const
        result.append(new_value)
    # t
    for _ in range(counts.t):
        result.append(np.zeros(counts.n_sample))

    return np.array(result).T
