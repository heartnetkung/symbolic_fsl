import pandas as pd
from ...graphic import *
from itertools import combinations
from math import ceil


def make_rel_df(grids: list[Grid], all_shapes: list[list[Shape]])->pd.DataFrame:
    result = {'sample_index': [], 'shape_index': [], 'rel': []}
    n_samples = len(all_shapes)

    for sample_id, shapes in enumerate(all_shapes):

        for shape_id, shape in enumerate(shapes):
            for prop in _gen_prop(shape):
                result['sample_index'].append(sample_id)
                result['shape_index'].append(shape_id)
                result['rel'].append(prop)

        for (shape_id1, shape_id2) in combinations(range(len(shapes)), 2):
            shape1, shape2 = shapes[shape_id1], shapes[shape_id2]
            for rel in _gen_rel(shape1, shape2)+_gen_rel(shape2, shape1):
                result['sample_index'].append(sample_id)
                result['shape_index'].append(shape_id1)
                result['rel'].append(rel)
    return pd.DataFrame(result)


def _gen_prop(a: Shape)->list[str]:
    return [
        f'x_{a.x}', f'y_{a.y}',
        f'x_mid_{a.x+ceil(a.width/2)}', f'y_mid_{a.y+ceil(a.height/2)}',
        f'w_{a.width}', f'h_{a.height}', f'mass_{a.mass}', f'top_color_{a.top_color}'
    ]


def _gen_rel(a: Shape, b: Shape)->list[str]:
    result = []
    if _is_contain(a, b):
        result.append('contain')
    elif _is_contain(b, a):
        result.append('contained')
    return result


def _is_contain(a: Shape, b: Shape)->bool:
    a_x2, a_y2 = a.x+a.width, a.y+a.height
    b_x2, b_y2 = b.x+b.width, b.y+b.height
    if ((a.x == b.x) and (a.y == b.y) and (a_x2 == b_x2) and (a_y2 == b_y2)):
        return False
    if ((a.x <= b.x) and (a.y <= b.y) and (a_x2 >= b_x2) and (a_y2 >= b_y2)):
        return True
    return False
