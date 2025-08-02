from ...base import *
from ...graphic import *
from ...ml import *
from copy import deepcopy
from ..util import *
from ...manager.task import PartitionlessLogicTask
from enum import Enum
from .apply_partitionless_logic import is_single_fullsize_shape, split_shapes_equally
from itertools import permutations


class ApplyPartitionlessUnion(ModelFreeArcAction[PartitionlessLogicTask]):
    def __init__(self, order: list[int], row_count: int,
                 col_count: int, params: GlobalParams)->None:
        self.order = order
        self.row_count = row_count
        self.col_count = col_count
        self.params = params
        super().__init__()

    def perform(self, state: ArcState,
                task: PartitionlessLogicTask)->Optional[ArcState]:
        new_out_shapes = self.apply(state)
        if not isinstance(state, ArcTrainingState):
            return state.update(out_shapes=new_out_shapes)

        return state.update(out_shapes=new_out_shapes,
                            reparse_count=self.params.max_reparse)

    def apply(self, state: ArcState)->Optional[list[list[Shape]]]:
        assert state.out_shapes is not None
        if not is_single_fullsize_shape(state.out_shapes, state.x):
            return None

        all_splitted_shapes = split_shapes_equally(
            state.out_shapes, self.row_count, self.col_count)
        if all_splitted_shapes is None:
            return None

        result, order_len = [], len(self.order)
        for shapes in all_splitted_shapes:
            if order_len != len(shapes):
                return None

            reordered_shapes = [shapes[i] for i in self.order]
            new_grid = draw_canvas(shapes[0].width, shapes[0].height,
                                   reordered_shapes, include_xy=False)
            result.append([Unknown(0, 0, new_grid)])
        return result
