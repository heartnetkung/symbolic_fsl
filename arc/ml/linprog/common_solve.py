import numpy as np
from .variables import *
from functools import cached_property
import re


EPSILON = 0.01
MAX_SOLVE = 10
# sometimes the optimal result for our "approximate" objective is not the same as
# the optimal result for the "actual" objective, so the full sweep is required.
COEF_SUM_THRESHOLD = 1-EPSILON
ANY_PATTERN = re.compile('.*')


class LinprogResult:
    '''Bundle class storing all optimizing variables'''

    def __init__(self, result: np.ndarray, counts: VariableCount)->None:
        len0, len1, len2 = counts.c0p, counts.c1p, counts.c2p

        # layout extraction
        c0p = result[:len0]
        c0m = result[len0:2*len0]
        c1p = result[2*len0: 2*len0+len1]
        c1m = result[2*len0+len1: 2*len0+2*len1]
        c2p = result[2*len0+2*len1: 2*len0+2*len1+len2]
        c2m = result[2*len0+2*len1+len2:2*len0+2*len1+2*len2]
        self.c0, self.c1, self.c2 = c0p-c0m, c1p-c1m, c2p-c2m
        self.poly_coef = np.array([*self.c0, *self.c1, *self.c2])
        self.fail_count = np.round(
            np.sum(result[counts.c_total:counts.c_total+counts.b]))
        self.has_c1 = np.logical_not(np.isclose(self.c1, 0, atol=EPSILON).all())
        self.has_c2 = np.logical_not(np.isclose(self.c2, 0, atol=EPSILON).all())

    def to_key(self)->str:
        return ' '.join([f'{coef:z.1f}'for coef in self.poly_coef])

    def __repr__(self)->str:
        return f'LinprogResult: {self.poly_coef}'

    @cached_property
    def score(self)->int:
        return (_coef_scoring(self.c0, 0) +
                _coef_scoring(self.c1, 1) +
                _coef_scoring(self.c2, 2))


def update_mask(mask: np.ndarray, counts: VariableCount,
                new_result: np.ndarray)->None:
    for i in range(counts.c0, counts.c_total):
        if not np.isclose(new_result[i], 0):
            mask[i] *= C_MAX


def _coef_scoring(coefs: np.ndarray, degree: int)->int:
    score = 0
    for coef in coefs:
        if np.allclose(coef, 0, atol=EPSILON):
            continue
        if not (np.allclose(coef, 1, atol=EPSILON) or
                np.allclose(coef, -1, atol=EPSILON)):
            score += 1
        score += degree
    return score
