from ...base import *
from ...graphic import *
from ...manager.reparse import *
from enum import Enum
from itertools import combinations
import networkx as nx
from typing import Generator
import numpy as np

NEARBY_THRESHOLD = 1


class MergeNearbyParam(Enum):
    skip = 0
    normal = 1


class MergeNearby(ModelFreeArcAction[MergeNearbyTask]):
    '''
    Look for y_shapes near the image's edge and check if they are partially visible.
    The source of full shapes come from x_shapes.
    '''

    def __init__(self, param: MergeNearbyParam = MergeNearbyParam.skip)->None:
        self.param = param
        super().__init__()

    def perform(self, state: ArcState, task: MergeNearbyTask)->Optional[ArcState]:
        assert state.x_shapes is not None
        if self.param == MergeNearbyParam.skip:
            return state

        new_x_shapes = self._reparse(state.x_shapes, state.x)
        if not isinstance(state, ArcTrainingState):
            return state.update(x_shapes=new_x_shapes, out_shapes=new_x_shapes)

        assert state.y_shapes is not None
        new_y_shapes = self._reparse(state.y_shapes, state.y)
        return state.update(
            x_shapes=new_x_shapes, out_shapes=new_x_shapes,
            y_shapes=new_y_shapes, reparse_count=state.reparse_count+1)

    def _reparse(self, all_shapes: list[list[Shape]],
                 grids: list[Grid])->list[list[Shape]]:
        all_new_shapes = []
        for shapes, grid in zip(all_shapes, grids):
            new_shapes, graph = [], NearbyGraph(shapes)
            for cluster in graph.connected_components():
                if len(cluster) == 1:
                    new_shapes.append(cluster.pop())
                else:
                    new_shapes.append(_merge_shapes(cluster, grid))
            all_new_shapes.append(new_shapes)
        return all_new_shapes


class NearbyGraph:
    def __init__(self, shapes: list[Shape])->None:
        self.graph = nx.Graph()
        edges = [(a, b) for a, b in combinations(shapes, 2) if _is_nearby(a, b)]
        self.graph.add_edges_from(edges)

    def connected_components(self)->Generator[set[Shape], None, None]:
        return nx.connected_components(self.graph)


def _merge_shapes(shapes: set[Shape], grid: Grid)->Shape:
    canvas = make_grid(grid.width, grid.height)
    for shape in shapes:
        shape.draw(canvas)
    x, y, shape_grid = trim(np.array(canvas.data))
    return Unknown(x, y, shape_grid)


def _is_nearby(a: Shape, b: Shape)->bool:
    x_a_range, x_b_range = range(a.x, a.x+a.width), range(b.x, b.x+b.width)
    if not _is_range_nearby(x_a_range, x_b_range):
        return False
    y_a_range, y_b_range = range(a.y, a.y+a.height), range(b.y, b.y+b.height)
    if not _is_range_nearby(y_a_range, y_b_range):
        return False
    return True


def _is_range_nearby(a: range, b: range)->bool:
    if a.start-b.stop > NEARBY_THRESHOLD:
        return False
    if b.start-a.stop > NEARBY_THRESHOLD:
        return False
    return True
