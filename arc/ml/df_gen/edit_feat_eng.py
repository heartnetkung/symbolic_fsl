from ...graphic import *
import pandas as pd
from typing import Optional
from ...constant import *


def edit_feat_eng(all_shapes: list[list[Shape]], index: int,
                  result: dict[str, list])->None:
    for shapes in all_shapes:
        for k, v in _edit_feat_eng(shapes, index).items():
            result[k] = result.get(k, [])+[v]


def _edit_feat_eng(shapes: list[Shape], index: int)->dict[str, int]:
    result: dict[str, int] = {}
    masses = [shape.mass for shape in shapes]
    result['mass_rank'] = to_rank(masses)[index]
    return result


def to_rank(values: list[int])->list[int]:
    lookup = {}
    for i, val in enumerate(sorted(set(values), reverse=True)):
        lookup[val] = i
    return [lookup[val] for val in values]
