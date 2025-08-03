from .variables import *
from .reg_constraints import *
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import OptimizeResult, milp
from typing import Optional, Pattern
from ...constant import GlobalParams
from .common_solve import *


def solve_reg(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
              lambda_: int = 50, max_result: int = 5,
              deg2_pattern: Pattern = ANY_PATTERN)->list[LinprogResult]:
    n_col, n_sample = X.shape[1], X.shape[0]
    assert n_col > 0
    assert n_sample == len(y)
    if MIN_MATCH > n_sample:
        return []

    deg2_cols = _make_deg2_cols(list(X.columns), params, deg2_pattern)
    counts = VariableCount(X, deg2_cols)
    mask = np.ones(counts.total)
    fail_count = -1

    ans, large_result_count = {}, 0
    for _ in range(MAX_SOLVE):
        raw_result = _solve(X, y, lambda_, mask, counts, params, deg2_cols)
        if not raw_result.success:
            break

        new_result = LinprogResult(raw_result.x, counts)
        if fail_count == -1:
            fail_count = new_result.fail_count
        elif fail_count < new_result.fail_count:
            break

        update_mask(mask, counts, raw_result.x)
        if not _is_overfit(counts, new_result):
            ans[new_result.to_key()] = new_result
            if abs(new_result.poly_coef.sum()) > COEF_SUM_THRESHOLD:
                large_result_count += 1
            if large_result_count == max_result:
                break

    return sorted(list(ans.values()), key=lambda x: x.score)


def make_reg_features(X: pd.DataFrame, params: GlobalParams,
                      deg2_pattern: Pattern = ANY_PATTERN)->np.ndarray:
    n_col, n_sample = X.shape[1], X.shape[0]
    result, columns = [], list(X.columns)

    # c0
    result.append([1]*n_sample)
    # c1
    for col in columns:
        result.append(X[col])
    # c2
    for col1, col2 in _make_deg2_cols(columns, params, deg2_pattern):
        result.append(X[col1]*X[col2])

    return np.array(result).T


def make_reg_columns(X: pd.DataFrame, params: GlobalParams,
                     deg2_pattern: Pattern = ANY_PATTERN)->list[str]:
    columns = list(X.columns)
    result = ['1']+columns
    for col1, col2 in _make_deg2_cols(columns, params, deg2_pattern):
        result.append(f'{col1}*{col2}')
    return result

# ========================
# private functions
# ========================


def _solve(X: pd.DataFrame, y: np.ndarray, lambda_: int, mask: np.ndarray,
           counts: VariableCount, params: GlobalParams,
           deg2_cols: list[tuple[str, str]])->OptimizeResult:
    variables = make_variables(counts, False, list(X.columns), lambda_)
    constraints = [make_min_match_constraints(counts),
                   *make_dataset_constraints(counts, X, y, deg2_cols),
                   *make_mistake_notone_constraints(counts)]
    return milp(np.array(variables.cost)*mask,
                integrality=np.array(variables.integrality),
                bounds=variables.bounds, constraints=constraints,
                options={'time_limit': params.linprog_time_limit})


def _is_overfit(counts: VariableCount, result: LinprogResult)->bool:
    successful_fit = counts.n_sample - result.fail_count
    assert successful_fit != 1
    if successful_fit == 3 and result.has_c2:
        return True
    if successful_fit == 2 and (result.has_c2 or result.has_c1):
        return True
    return False


def _make_deg2_cols(col_names: list[str], params: GlobalParams,
                    deg2_pattern: Pattern)->list[tuple[str, str]]:
    if not params.ppdt_enable_deg2:
        return []
    result = []
    for col1, col2 in combinations_with_replacement(col_names, 2):
        if ((deg2_pattern.match(col1) is not None) or
                (deg2_pattern.match(col2) is not None)):
            result.append((col1, col2))
    return result
