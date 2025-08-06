from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from copy import deepcopy
from ..util import *
from enum import Enum
from .fitb_generate_df import *


class ExpansionMode(Enum):
    top_left = 0
    top_right = 1
    bottom_left = 2
    bottom_right = 3
    center = 4

    def _get_offset(self, diff_width: int, diff_height: int)->Optional[tuple[int, int]]:
        if diff_width < 0 or diff_height < 0:
            return None
        if self == ExpansionMode.top_left:
            return 0, 0
        if self == ExpansionMode.top_right:
            return diff_width, 0
        if self == ExpansionMode.bottom_left:
            return 0, diff_height
        if self == ExpansionMode.bottom_right:
            return diff_width, diff_height
        if self == ExpansionMode.center:
            offset_x, offset_y = diff_width/2, diff_height/2
            if not np.allclose([offset_x % 1, offset_y % 1], 0):
                return None
            return int(offset_x), int(offset_y)
        raise Exception('unknown mode')

    def get_bound(self, shape: Shape, width: int, height: int)->Optional[
            tuple[int, int, int, int]]:
        diff_width, diff_height = width-shape.width, height-shape.height
        offsets = self._get_offset(diff_width, diff_height)
        if offsets is None:
            return None
        return (offsets[0], offsets[0]+shape.width,
                offsets[1], offsets[1]+shape.height)

    def expand(self, shape: Shape, width: int, height: int)->Optional[Unknown]:
        diff_width, diff_height = width-shape.width, height-shape.height
        offsets = self._get_offset(diff_width, diff_height)
        if offsets is None:
            return None

        result = make_grid(width, height)
        Unknown(offsets[0], offsets[1], shape._grid).draw(result)
        return Unknown(shape.x-offsets[0], shape.y-offsets[1], result)


class FillInTheBlank(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, expansion: ExpansionMode, feat_index: int,
                 width_model: MLModel, height_model: MLModel, pixel_model: MLModel,
                 params: GlobalParams, gen_df: bool = True)->None:
        self.expansion = expansion
        self.feat_index = feat_index
        self.width_model = width_model
        self.height_model = height_model
        self.pixel_model = pixel_model
        self.params = params
        self.gen_df = gen_df
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes is not None

        atn = task.atn
        shape_df = default_make_df(state, atn, self.feat_index)
        new_widths = self.width_model.predict_int(shape_df)
        new_heights = self.height_model.predict_int(shape_df)
        result = deepcopy(state.out_shapes)
        grids = get_grids(state, atn)

        for width, height, grid, id1, shape_ids in zip(
                new_widths, new_heights, grids, atn.sample_index, atn.x_index):

            id2 = shape_ids[self.feat_index]
            shape = result[id1][id2]
            new_shape = self.expansion.expand(shape, width, height)
            if new_shape is None:
                return None

            bound = self.expansion.get_bound(shape, width, height)
            assert bound is not None

            if self.gen_df:
                pixel_df = generate_pixel_df([grid], [new_shape], [bound])
            else:
                pixel_df = _gen_dummy_df(new_shape._grid)

            pixel_color = self.pixel_model.predict_int(pixel_df)
            for x, y, color in zip(pixel_df['x'], pixel_df['y'], pixel_color):
                new_shape.grid.safe_assign(x, y, color)

            result[id1][id2] = new_shape
        return state.update(out_shapes=result)

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.width_model, MemorizedModel)
        assert isinstance(self.height_model, MemorizedModel)
        assert isinstance(self.pixel_model, StepMemoryModel)

        shape_df = default_make_df(state, task.atn, self.feat_index)
        widths, heights = self.width_model.result, self.height_model.result
        w_models = regressor_factory(shape_df, widths, self.params, 'fitb.w')
        h_models = regressor_factory(shape_df, heights, self.params, 'fitb.h')

        x_shapes = get_x_col(state, task.atn, self.feat_index)
        expanded_shapes, bounds = [], []
        for shape, w, h in zip(x_shapes, widths, heights):
            expanded_shape = self.expansion.expand(shape, w, h)
            if expanded_shape is None:
                return []

            bound = self.expansion.get_bound(shape, w, h)
            assert bound is not None
            expanded_shapes.append(expanded_shape)
            bounds.append(bound)

        grids = get_grids(state, task.atn)
        pixel_df = generate_pixel_df(grids, expanded_shapes, bounds)
        p_models = regressor_factory(
            pixel_df, self.pixel_model.result, self.params, 'fitb.p')

        return [FillInTheBlank(
            self.expansion, self.feat_index, w_model, h_model, p_model, self.params)
            for w_model, h_model, p_model in model_selection(
                w_models, h_models, p_models)]


def _gen_dummy_df(grid: Grid)->pd.DataFrame:
    data = {'x': [], 'y': []}
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.data[y][x] == NULL_COLOR:
                data['x'].append(x)
                data['y'].append(y)
    return pd.DataFrame(data)
