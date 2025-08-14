from ...base import *
from ...graphic import *
from ...ml import *
from copy import deepcopy
from ..util import *
from ...manager.draw_line import *
from ...manager.task import TrainingDrawLineTask, DrawLineTask
import numpy as np
from .draw_line_df import *


class DrawLine(ModelBasedArcAction[TrainingDrawLineTask, DrawLineTask]):
    def __init__(self, init_x_model: MLModel, init_y_model: MLModel, dir_model: MLModel,
                 nav_model: MLModel, color_model: MLModel, params: GlobalParams)->None:
        self.init_x_model = init_x_model
        self.init_y_model = init_y_model
        self.dir_model = dir_model
        self.color_model = color_model
        self.nav_model = nav_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: DrawLineTask)->Optional[ArcState]:
        assert state.out_shapes is not None

        atn = task.atn
        shape_df = default_make_df(state, task)
        grids = get_grids(state, atn)
        new_out_shapes = deepcopy(state.out_shapes)
        x_values = self.init_x_model.predict_int(shape_df)
        y_values = self.init_y_model.predict_int(shape_df)
        color_values = self.color_model.predict_int(shape_df)
        dir_values = self.dir_model.predict_enum(shape_df, Direction)

        for grid, x, y, dir_, color, id1 in zip(
                grids, x_values, y_values, dir_values, color_values, atn.sample_index):
            line = _make_line(grid, x, y, dir_, self.nav_model, color)
            if line is None:
                return None

            new_out_shapes[id1].append(line.to_shape())

        if isinstance(state, ArcTrainingState):
            return state.update(out_shapes=deduplicate_all_shapes(new_out_shapes),
                                # needed because y is reparsed
                                y_shapes=task.all_lines.to_y_shapes())
        return state.update(out_shapes=deduplicate_all_shapes(new_out_shapes))

    def train_models(self, state: ArcTrainingState,
                     task: TrainingDrawLineTask)->list[InferenceAction]:
        assert isinstance(self.init_x_model, MemorizedModel)
        assert isinstance(self.init_y_model, MemorizedModel)
        assert isinstance(self.dir_model, MemorizedModel)
        assert isinstance(self.color_model, MemorizedModel)
        assert isinstance(self.nav_model, StepMemoryModel)

        shape_df = default_make_df(state, task)
        pixel_df = make_training_nav_df(state, task)
        if pixel_df is None:
            return []

        labels = [self.init_x_model.result, self.init_y_model.result,
                  self.color_model.result, self.dir_model.result]
        label_types = [LabelType.reg]*3+[LabelType.cls_]
        all_models = make_all_models(
            shape_df, self.params, 'draw_line', labels, label_types)

        n_models = make_classifier(
            pixel_df, self.nav_model.result, self.params, 'draw.n')
        all_models.append(n_models)

        return [DrawLine(x_model, y_model, d_model, n_model, c_model, self.params)
                for x_model, y_model, c_model, d_model, n_model in model_selection(
            *all_models)]


def _make_line(full_grid: Grid, init_x: int, init_y: int, init_dir: Direction,
               nav_model: MLModel, color: int)->Optional[Line]:
    temp_grid = Grid(deepcopy(full_grid.data))
    temp_grid.data[init_y][init_x] = color
    coords = [Coordinate(init_x, init_y)]
    current_coord, current_dir = Coordinate(init_x, init_y), init_dir
    normal_break = False

    for _ in range(MAX_LINE_LENGTH):
        df = generate_step_df(temp_grid, current_coord, current_dir, color)
        nav = nav_model.predict_enum(df, Navigation)[0]
        if nav == Navigation.stop:
            normal_break = True
            break
        if nav == Navigation.turn_left:
            current_dir = current_dir.left()
        elif nav == Navigation.turn_right:
            current_dir = current_dir.right()

        current_coord = current_dir.proceed(current_coord)
        coords.append(current_coord)
        temp_grid.safe_assign_c(current_coord, color)

    if not normal_break:
        return None
    return Line.make(coords, color)
