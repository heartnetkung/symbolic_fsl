from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from ...attention import *
from copy import deepcopy
import pandas as pd
from ..util import *
from enum import Enum
import itertools


class Grouping(Enum):
    none = 0
    shape_color = 1
    shape = 2

    def get_key(self, shape: Shape)->Any:
        if self == Grouping.none:
            return id(shape)
        elif self == Grouping.shape_color:
            return repr(shape._grid.data)
        elif self == Grouping.shape:
            return repr(shape._grid.normalize_color().data)
        else:
            raise Exception('unknown')


class SelectOne(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, selection_model: MLModel, grouping: Grouping,
                 params: GlobalParams)->None:
        self.grouping = grouping
        self.selection_model = selection_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes != None

        all_groups, all_ranks, sample_index = extract_group(state, self.grouping)
        df = _make_df(all_groups, all_ranks)
        selections = self.selection_model.predict_bool(df)
        result = [[] for _ in range(len(state.out_shapes))]

        for id1, selection, groups in zip(sample_index, selections, all_groups):
            if selection:
                new_shape = deepcopy(groups[0])
                new_shape.x = 0
                new_shape.y = 0
                result[id1].append(new_shape)

        for shapes in result:
            if len(shapes) != 1:
                return None
        return state.update(out_shapes=result)

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.selection_model, MemorizedModel)

        all_groups, all_ranks, sample_index = extract_group(state, self.grouping)
        df = _make_df(all_groups, all_ranks)
        all_models = make_classifier(
            df, self.selection_model.result, self.params, 'select_one')
        return [SelectOne(model, self.grouping, self.params) for model in all_models]


def _make_df(all_shapes: list[list[Shape]],
             all_ranks: list[list[float]])->pd.DataFrame:
    df = generate_df(all_shapes=all_shapes)
    df['group_rank'] = list(itertools.chain(*all_ranks))
    return df


def extract_group(state: ArcState, grouping: Grouping)->tuple[
        list[list[Shape]], list[list[float]], list[int]]:
    assert state.out_shapes != None
    result, group_rank, sample_index = [], [], []

    for sample_id, shapes in enumerate(state.out_shapes):
        group = {}
        for shape in shapes:
            key = grouping.get_key(shape)
            if key in group:
                group[key].append(shape)
            else:
                group[key] = [shape]

        for values in group.values():
            result.append([values[0]])
            sample_index.append(sample_id)
        counts = [len(values) for values in group.values()]
        group_rank.append(to_rank_float(counts))
    return result, group_rank, sample_index
