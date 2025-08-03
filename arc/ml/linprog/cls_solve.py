from scipy.optimize import OptimizeResult, milp
from typing import Optional, Pattern
from ...constant import GlobalParams
from .common_solve import *
from dataclasses import dataclass
from scipy.optimize import LinearConstraint
from collections.abc import Iterable

# our dataset is mostly integers, thus EPSILON == 1 is fine
# the upside is that it produces more sensible coeficients
UNIT_EPSILON = 1


def solve_cls(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
              max_result: int = 5, scoring_threshold: int = 5)->list[LinprogResult]:
    n_col, n_sample = X.shape[1], X.shape[0]
    assert n_col > 0
    assert n_sample == len(y)

    counts = VariableCount(X, [], True)
    mask = np.ones(counts.total)

    ans, large_result_count = {}, 0
    for _ in range(MAX_SOLVE):
        raw_result = _solve(X, y, mask, counts, params)
        if not raw_result.success:
            break

        new_result = LinprogResult(raw_result.x, counts)
        update_mask(mask, counts, raw_result.x)
        ans[new_result.to_key()] = new_result
        if abs(new_result.poly_coef.sum()) > COEF_SUM_THRESHOLD:
            large_result_count += 1
        if large_result_count == max_result:
            break
    return _postprocess(ans.values(), scoring_threshold)


def _postprocess(results: Iterable[LinprogResult], threshold: int)->list[LinprogResult]:
    return sorted(filter(lambda x: x.score <= threshold, results),
                  key=lambda x: x.score)


def _solve(X: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
           counts: VariableCount, params: GlobalParams)->OptimizeResult:
    variables = make_variables(counts, True, list(X.columns))
    return milp(np.array(variables.cost)*mask,
                integrality=np.array(variables.integrality),
                bounds=variables.bounds,
                constraints=_make_cls_constraints(counts, X, y),
                options={'time_limit': params.linprog_time_limit})


def _make_cls_constraints(counts: VariableCount, X: pd.DataFrame,
                          y: np.ndarray)->LinearConstraint:
    '''Match as many data points as possible.'''
    feats = _make_internal_features(counts, X)
    ub = np.where(y, np.inf, -UNIT_EPSILON)
    lb = np.where(y, 0, -np.inf)
    return LinearConstraint(feats, ub=ub, lb=lb)  # type:ignore


def _make_internal_features(counts: VariableCount, X: pd.DataFrame)->np.ndarray:
    result, columns = [], list(X.columns)
    # c0p, c0m
    result.append([1]*counts.n_sample)
    result.append([-1]*counts.n_sample)
    # c1p, c1m
    for col in columns:
        result.append(X[col])
    for col in columns:
        result.append(-X[col])
    return np.array(result).T


def make_cls_features(X: pd.DataFrame, params: GlobalParams)->np.ndarray:
    # c0
    result = [[1]*X.shape[0]]
    # c1
    for col in X.columns:
        result.append(X[col])  # type:ignore
    return np.array(result).T


def make_cls_columns(X: pd.DataFrame, params: GlobalParams)->list[str]:
    return ['1']+list(X.columns)
