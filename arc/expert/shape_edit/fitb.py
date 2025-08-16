from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from copy import deepcopy
from ..util import *
from enum import Enum
from .fitb_generate_df import *
from .fitb_expansion import ExpansionMode


class FillInTheBlank(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, expansion: ExpansionMode, feat_index: int,
                 width_model: MLModel, height_model: MLModel, pixel_model: MLModel,
                 params: GlobalParams)->None:
        self.expansion = expansion
        self.feat_index = feat_index
        self.width_model = width_model
        self.height_model = height_model
        self.pixel_model = pixel_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes is not None

        atn = task.atn
        result = deepcopy(state.out_shapes)
        grids = get_grids(state, atn)
        shape_df = default_make_df(state, task, self.feat_index)

        if self.expansion.is_symmetry():
            x_shapes = get_x_col(state, atn, self.feat_index)
            new_widths, new_heights = self.expansion.get_widths_heights(x_shapes)
        else:
            new_widths = self.width_model.predict_int(shape_df)
            new_heights = self.height_model.predict_int(shape_df)

        for width, height, grid, id1, shape_ids, record in zip(
                new_widths, new_heights, grids, atn.sample_index, atn.x_index,
                shape_df.to_dict('records')):

            id2 = shape_ids[self.feat_index]
            shape = result[id1][id2]
            bound = self.expansion.get_bound(shape, width, height)
            if bound is None:
                return None

            new_shape = draw_shape(shape, bound, width, height)
            pixel_df = generate_pixel_df([grid], [new_shape], [bound], [record])
            if len(pixel_df) == 0:
                return None

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

        x_shapes = get_x_col(state, task.atn, self.feat_index)
        shape_df = default_make_df(state, task, self.feat_index)

        if self.expansion.is_symmetry():
            widths, heights = self.expansion.get_widths_heights(x_shapes)
            all_models: list[list[MLModel]] = [[ConstantModel(99)]]*2
        else:
            widths, heights = self.width_model.result, self.height_model.result
            labels = [widths, heights]
            label_types = [LabelType.reg]*2
            all_models: list[list[MLModel]] = make_all_models(
                shape_df, self.params, 'fitb.size', labels, label_types)

        expanded_shapes, bounds = [], []
        for shape, w, h in zip(x_shapes, widths, heights):
            bound = self.expansion.get_bound(shape, w, h)
            if bound is None:
                return []

            expanded_shape = draw_shape(shape, bound, w, h)
            expanded_shapes.append(expanded_shape)
            bounds.append(bound)

        grids = get_grids(state, task.atn)
        pixel_df = generate_pixel_df(
            grids, expanded_shapes, bounds, shape_df.to_dict('records'))
        p_models = make_regressor(
            pixel_df, self.pixel_model.result, self.params, 'fitb.p')
        all_models.append(p_models)

        return [FillInTheBlank(
            self.expansion, self.feat_index, w_model, h_model, p_model, self.params)
            for w_model, h_model, p_model in model_selection(*all_models)]


def draw_shape(shape: Shape, bound: tuple[int, int, int, int],
               width: int, height: int)->Unknown:
    x_min, x_max, y_min, y_max = bound
    result = make_grid(width, height)
    Unknown(x_min, y_min, shape._grid).draw(result)
    return Unknown(shape.x-x_min, shape.y-y_min, result)
