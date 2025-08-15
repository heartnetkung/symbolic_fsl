from ...base import ArcTrainingState
from .shape_graph import ShapeGraph
from .list_reparse_relationship import *
from itertools import product, combinations
from dataclasses import dataclass


@dataclass(frozen=True)
class ReparseEdgeTask(ModelFreeTask):
    supershape: ShapeGraph
    transformed_supershape: ShapeGraph
    colorless_supershape: ShapeGraph
    colorless_transformed_supershape: ShapeGraph


def create_reparse_edge(state: ArcTrainingState)->ReparseEdgeTask:
    assert state.x_shapes is not None
    assert state.y_shapes is not None
    all_edges = _create_subset_edges(state.y_shapes, state.x_shapes, False)
    all_nodes = [state.x_shapes, state.y_shapes]
    return ReparseEdgeTask(
        ShapeGraph(all_nodes, all_edges['subshape'], True),  # type:ignore
        ShapeGraph(all_nodes, all_edges['transformed_subshape'], True),  # type:ignore
        ShapeGraph(all_nodes, all_edges['colorless_subshape'], True),  # type:ignore
        ShapeGraph(all_nodes,
                   all_edges['colorless_transformed_subshape'], True))  # type:ignore


def _create_subset_edges(
        from_all_shapes: list[list[Shape]], to_all_shapes: list[list[Shape]],
        subshape: bool)->dict[str, list[tuple[int, int]]]:
    result = {'subshape': [], 'transformed_subshape': [], 'approx_subshape': [],
              'colorless_subshape': [], 'colorless_transformed_subshape': []}
    for from_shapes, to_shapes in zip(from_all_shapes, to_all_shapes):
        for from_shape, to_shape in product(from_shapes, to_shapes):
            from_id, to_id = id(from_shape), id(to_shape)
            new_edge = (to_id, from_id) if subshape else (from_id, to_id)
            rel = find_subshape(from_shape, to_shape)
            for result_key, result_value in result.items():
                if result_key in rel:
                    result_value.append(new_edge)
    return result
