from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import pandas as pd
from ..util import *
from copy import deepcopy


class ApplyLogic(ModelFreeArcAction[AttentionTask]):
    def __init__(self, feat_indexes: list[int], color: int, type: LogicType)->None:
        assert len(feat_indexes) > 1
        for index in feat_indexes:
            assert index >= 0

        self.feat_indexes = feat_indexes
        self.color = color
        self.type = type
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes is not None

        all_shapes, atn = deepcopy(state.out_shapes), task.atn
        to_remove = set()

        for id1, indexes in zip(atn.sample_index, atn.x_index):
            id2s = [indexes[feat_index] for feat_index in self.feat_indexes]
            new_shape = all_shapes[id1][id2s[0]]
            for id2 in id2s[1:]:
                new_shape = apply_logic(
                    new_shape, all_shapes[id1][id2], self.color, self.type)
            all_shapes[id1].append(new_shape)

            for id2 in id2s:
                to_remove.add((id1, id2))

        for i, shapes in enumerate(all_shapes):
            all_shapes[i] = [shape for j, shape in enumerate(shapes)
                             if (i, j) not in to_remove]
        return state.update(out_shapes=deduplicate_all_shapes(all_shapes))

    def apply(self, state: ArcState, atn: Attention)->list[Shape]:
        assert state.out_shapes is not None

        result, all_shapes = [], state.out_shapes
        for id1, indexes in zip(atn.sample_index, atn.x_index):
            id2s = [indexes[feat_index] for feat_index in self.feat_indexes]
            new_shape = all_shapes[id1][id2s[0]]
            for id2 in id2s[1:]:
                new_shape = apply_logic(
                    new_shape, all_shapes[id1][id2], self.color, self.type)
            result.append(new_shape)
        return result
