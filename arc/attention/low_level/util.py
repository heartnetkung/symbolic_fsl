from dataclasses import dataclass
from ...graphic import *
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd


@dataclass(frozen=True, order=True)
class ShapeId:
    sample_id: int
    shape_index: int


def resolve_all_shapes(all_shape_ids: list[list[ShapeId]],
                       all_shapes: list[list[Shape]])->list[list[Shape]]:
    return [resolve_shapes(shape_ids, all_shapes) for shape_ids in all_shape_ids]


def resolve_shapes(shape_ids: list[ShapeId],
                   all_shapes: list[list[Shape]])->list[Shape]:
    return [lookup(shape_id, all_shapes) for shape_id in shape_ids]


def resolve_all_grids(all_shape_ids: list[list[ShapeId]],
                      grids: list[Grid])->list[Grid]:
    return [grids[shape_ids[0].sample_id] for shape_ids in all_shape_ids]


def resolve_grids(shape_ids: list[ShapeId],
                  grids: list[Grid])->list[Grid]:
    return [grids[shape_id.sample_id] for shape_id in shape_ids]


def lookup(id_: ShapeId, all_shapes: list[list[Shape]])->Shape:
    return all_shapes[id_.sample_id][id_.shape_index]


def to_shape_ids(all_shapes: list[list[Shape]])->list[ShapeId]:
    result = []
    for sample_id, shapes in enumerate(all_shapes):
        for shape_index, shape in enumerate(shapes):
            result.append(ShapeId(sample_id, shape_index))
    return result


def sort_property(identifiers: list[ShapeId],
                  all_shapes: list[list[Shape]])->list[ShapeId]:
    def key_func(id_: ShapeId)->float:
        shape = lookup(id_, all_shapes)
        return shape.x*1e6+shape.y*1e4+shape.width*1e2+shape.height
    return sorted(identifiers, key=key_func)


def to_distance_matrix(df: pd.DataFrame, metric: str)->np.ndarray:
    # There are many metric types as listed below
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    # [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].

    # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    # [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’,
    # ‘hamming’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’,
    # ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
    # ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    return pairwise_distances(df, metric=metric)
