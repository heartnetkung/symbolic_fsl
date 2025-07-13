from ...graphic import *
from ..low_level import *
import pandas as pd
import numpy as np
from itertools import permutations
from dataclasses import dataclass
from operator import itemgetter
from typing import Optional
import math

MAX_PERMUTATION = 500


def gen_all_possible_index(
    all_shapes: list[list[Shape]], arity: list[int])->Optional[
        tuple[list[int], list[list[int]]]]:

    sample_index, x_index, total_arity = [], [], sum(arity)
    for sample_id, shapes in enumerate(all_shapes):
        if math.perm(len(shapes), total_arity) > MAX_PERMUTATION:
            return None

        for combination in permutations(range(len(shapes)), total_arity):
            new_index = list(combination)
            if new_index == do_align(arity, shapes, new_index):
                x_index.append(new_index)
                sample_index.append(sample_id)
    return sample_index, x_index


def do_align(arity: list[int], shapes: list[Shape], x_index: list[int])->list[int]:
    def key_func(index: int)->float:
        shape = shapes[index]
        return shape.x*1e6+shape.y*1e4+shape.width*1e2+shape.height

    result, offset = [], 0
    for arity_count in arity:
        if arity_count > 1:
            subindex = x_index[offset:offset+arity_count]
            result += sorted(subindex, key=key_func)
        elif arity_count == 1:
            result.append(x_index[offset])
        offset += arity_count
    return result
