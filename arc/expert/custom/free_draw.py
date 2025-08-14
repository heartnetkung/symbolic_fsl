from ...base import *
from ...graphic import *
from ...ml import *
from copy import deepcopy
from ..util import *
from ...manager.task import FreeDrawTask
from enum import Enum
from .free_draw_gen_df import generate_pixel_df
from .free_draw_feat_eng import cal_tile


class FreeDrawParam(Enum):
    skip = 0
    normal = 1


class FreeDraw(ModelBasedArcAction[FreeDrawTask, FreeDrawTask]):
    def __init__(self, param: FreeDrawParam, width_model: MLModel,
                 height_model: MLModel, pixel_model: MLModel,
                 params: GlobalParams)->None:
        self.param = param
        self.width_model = width_model
        self.height_model = height_model
        self.pixel_model = pixel_model
        self.params = params.update(ppdt_decision_tree_depth=3,
                                    ppdt_max_nested_regressors=5)

    def perform(self, state: ArcState, task: FreeDrawTask)->Optional[ArcState]:
        assert state.out_shapes is not None
        if self.param == FreeDrawParam.skip:
            return state

        df = generate_size_df(state.x, state.out_shapes)
        if df is None:
            return None

        widths = self.width_model.predict_int(df)
        heights = self.height_model.predict_int(df)
        new_out_shapes = []

        for grid, shapes, w, h in zip(state.x, state.out_shapes, widths, heights):
            shape = shapes[0]
            out_canvas = make_grid(w, h)

            pixel_df = generate_pixel_df([grid], [[shape]], [w], [h])
            pixel_color = self.pixel_model.predict_int(pixel_df)
            for x, y, color in zip(pixel_df['x'], pixel_df['y'], pixel_color):
                out_canvas.safe_assign(x, y, color)
            new_out_shapes.append([Unknown(shape.x, shape.y, out_canvas)])

        if not isinstance(state, ArcTrainingState):
            return state.update(out_shapes=new_out_shapes)
        return state.update(out_shapes=new_out_shapes,
                            reparse_count=self.params.max_reparse)

    def train_models(self, state: ArcTrainingState,
                     task: FreeDrawTask)->list[InferenceAction]:
        assert state.out_shapes is not None
        if self.param == FreeDrawParam.skip:
            return [self]

        assert isinstance(self.pixel_model, StepMemoryModel)
        df = generate_size_df(state.x, state.out_shapes)
        if df is None:
            return []

        widths = self.width_model.predict_int(df)
        heights = self.height_model.predict_int(df)
        df = generate_pixel_df(state.x, state.out_shapes, widths, heights)
        models = make_regressor(
            df, self.pixel_model.result, self.params, 'free_draw')
        return [FreeDraw(self.param, self.width_model, self.height_model,
                         model, self.params) for model in models]


def generate_size_df(
        grids: list[Grid], all_shapes: list[list[Shape]])->Optional[pd.DataFrame]:
    result = {'grid_width': [], 'grid_height': [], 'shape_width': [],
              'shape_height': [], 'tile_width': [], 'tile_height': []}
    for grid, shapes in zip(grids, all_shapes):
        if len(shapes) != 1:
            return None

        result['grid_width'].append(grid.width)
        result['grid_height'].append(grid.height)
        result['shape_width'].append(shapes[0].width)
        result['shape_height'].append(shapes[0].height)

        tile = cal_tile(shapes[0]._grid, True)
        if tile is not None:
            result['tile_width'].append(tile.width)
            result['tile_height'].append(tile.height)

    return pd.DataFrame(ensure_size(result))
