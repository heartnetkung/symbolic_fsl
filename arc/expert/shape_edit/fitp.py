from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from copy import deepcopy
from ..util import *
from .fitb import FillInTheBlank
from itertools import product
from typing import Iterable


class FITP(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    '''Fill in the patch.'''

    def __init__(self, fitb_action: FillInTheBlank, patch_model: MLModel)->None:
        self.fitb = fitb_action
        self.patch_model = patch_model
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        atn = task.atn
        df = default_make_df(state, atn, self.fitb.feat_index)
        patch_colors = self.patch_model.predict_int(df)

        # preprocess
        blob = _preprocess(state, task.atn, patch_colors, self.fitb.feat_index)
        if blob is None:
            return None
        bounds, replaced_shapes = blob

        # reuse fitb
        replaced_state = state.update(out_shapes=replaced_shapes)
        raw_output_state = self.fitb.perform(replaced_state, task)
        if raw_output_state is None:
            return None

        # postprocess
        assert raw_output_state.out_shapes is not None
        cropped_shapes = deepcopy(raw_output_state.out_shapes)
        for id1, shape_ids, bound in zip(atn.sample_index, atn.x_index, bounds):
            id2 = shape_ids[self.fitb.feat_index]
            original_shape = raw_output_state.out_shapes[id1][id2]
            cropped_shapes[id1][id2] = Unknown(
                original_shape.x, original_shape.y,
                original_shape._grid.crop(bound[0], bound[1], bound[2], bound[3]))

        return raw_output_state.update(out_shapes=cropped_shapes)

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.patch_model, MemorizedModel)
        blob = _preprocess(state, task.atn, self.patch_model.result,
                           self.fitb.feat_index)
        if blob is None:
            return []

        _, replaced_shapes = blob
        subactions = self.fitb.train_models(
            state.update(out_shapes=replaced_shapes), task)

        shape_df = default_make_df(state, task.atn, self.fitb.feat_index)
        patch_models = regressor_factory(shape_df, self.patch_model.result,
                                         self.fitb.params, 'fitp')
        return [FITP(subaction, patch_model)  # type:ignore
                for subaction, patch_model in product(subactions, patch_models)]


def _preprocess(state: ArcState, atn: Attention,
                patch_colors: Iterable[int], feat_index: int)->Optional[tuple[
        list[tuple[int, int, int, int]], list[list[Shape]]]]:
    assert state.out_shapes is not None
    replaced_shapes, bounds = deepcopy(state.out_shapes), []

    for id1, shape_ids, color in zip(atn.sample_index, atn.x_index, patch_colors):
        id2 = shape_ids[feat_index]
        original_shape = state.out_shapes[id1][id2]
        bound = find_bound(original_shape._grid, color)
        if bound is None:
            return None

        bounds.append(bound)
        replaced_shapes[id1][id2] = Unknown(
            original_shape.x, original_shape.y,
            original_shape._grid.replace_color(color, NULL_COLOR))
    return bounds, replaced_shapes


def find_bound(grid: Grid, color: int)->Optional[tuple[int, int, int, int]]:
    first_index = _index_of(grid, color, False)
    last_index = _index_of(grid, color, True)
    if (first_index is None) or (last_index is None):
        return None

    x_min, y_min = first_index
    x_max, y_max = last_index
    subgrid = grid.crop(x_min, y_min, x_max-x_min+1, y_max-y_min+1)
    if not FilledRectangle.is_valid(0, 0, subgrid):
        return None
    return x_min, y_min, x_max-x_min+1, y_max-y_min+1


def _index_of(grid: Grid, color: int, last: bool)->Optional[tuple[int, int]]:
    y_range = range(grid.height-1, -1, -1) if last else range(grid.height)
    x_range = range(grid.width-1, -1, -1) if last else range(grid.width)
    for y in y_range:
        for x in x_range:
            if grid.data[y][x] == color:
                return (x, y)
    return None
