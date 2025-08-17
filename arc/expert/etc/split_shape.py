from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import pandas as pd
from ..util import *
from copy import deepcopy
from ...algorithm.recursive_shape_split import recursive_shape_split


class SplitShape(ModelFreeArcAction[AttentionTask]):
    def __init__(self, feat_index: int)->None:
        self.feat_index = feat_index
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes is not None
        atn = task.atn
        if len(task.common_y_shapes) == 0:
            return None

        x_index = [index[self.feat_index] for index in atn.x_index]
        new_out_shapes = self._split_shape(
            atn.sample_index, x_index, state.out_shapes, task.common_y_shapes)
        if new_out_shapes is None:
            return None

        if not isinstance(state, ArcTrainingState):
            return state.update(out_shapes=new_out_shapes)

        assert state.y_shapes is not None
        new_y_shapes = self._split_shape(
            atn.sample_index, task.atn.y_index, state.y_shapes,  # type:ignore
            task.common_y_shapes)
        if new_y_shapes is None:
            return None

        return state.update(out_shapes=new_out_shapes, y_shapes=new_y_shapes,
                            attention_cache=None)

    def _split_shape(self, sample_index: list[int], shape_index: list[int],
                     all_shapes: list[list[Shape]],
                     subshapes: tuple[Shape, ...])->Optional[list[list[Shape]]]:
        seen_shapes, appending = set(), {}
        dedup = Deduplicator()

        for id1, id2 in zip(sample_index, shape_index):
            if dedup.has_seen_before((id1, id2)):
                continue

            container = all_shapes[id1][id2]
            new_shapes = recursive_shape_split(container, subshapes, True, True)
            if new_shapes is None:
                return None
            appending[id1] = appending.get(id1, [])+new_shapes
            seen_shapes.add((id1, id2))

        result = []
        for id1, shapes in enumerate(all_shapes):
            new_row = [shape for id2, shape in enumerate(shapes)
                       if (id1, id2) not in seen_shapes] + appending.get(id1, [])
            result.append(new_row)
        return result
