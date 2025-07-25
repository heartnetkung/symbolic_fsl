from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import pandas as pd
from ..util import *
from copy import deepcopy
import itertools


class Crop(ModelBasedArcAction[CropTask, CropTask]):
    def __init__(self, corner_finder: MLModel, params: GlobalParams,
                 include_corner: bool = True)->None:
        self.corner_finder = corner_finder
        self.include_corner = include_corner
        self.params = params

    def perform(self, state: ArcState, task: CropTask)->Optional[ArcState]:
        assert state.out_shapes != None
        new_out_shapes = []
        for canvas, x_grid in zip(get_canvases(state, task), state.x):
            df = _make_df([canvas], [x_grid])
            if df is None:
                return None

            corners = self.corner_finder.predict_bool(df)
            if np.sum(corners) != 4:
                return None

            correct_df = df[corners]
            assert isinstance(correct_df, pd.DataFrame)
            new_grid = _crop(canvas, correct_df, self.include_corner)
            new_out_shapes.append([Unknown(0, 0, new_grid)])

        if isinstance(state, ArcTrainingState):
            return state.update(out_shapes=new_out_shapes,
                                # needed because y is reparsed
                                y_shapes=new_out_shapes)
        return state.update(out_shapes=new_out_shapes)

    def train_models(self, state: ArcTrainingState,
                     task: CropTask)->list[InferenceAction]:
        # check include_corner
        assert state.out_shapes is not None
        assert isinstance(self.corner_finder, StepMemoryModel)

        canvases = get_canvases(state, task)
        df = _make_df(canvases, state.x)
        if df is None:
            return []

        models = classifier_factory(df, self.corner_finder.result, self.params, 'crop')
        return [Crop(model, self.params, self.include_corner) for model in models]


def get_canvases(state: ArcState, task: CropTask)->list[Grid]:
    if task.crop_from_out_shapes:
        assert state.out_shapes is not None
        assert state.x_bg is not None
        return [draw_canvas(grid.width, grid.height, shapes, bg)
                for grid, shapes, bg in zip(state.x, state.out_shapes, state.x_bg)]
    return state.x


def _make_df(grids: list[Grid], x_grids: list[Grid])->Optional[pd.DataFrame]:
    if len(grids) == 0:
        return None

    grid_data_table = generate_df(grids).to_dict('records')
    result = {col: [] for col in grid_data_table[0]}
    result |= {'cell(x,y)': [], 'cell(x-1,y)': [], 'cell(x,y-1)': [],
               'cell(x+1,y)': [], 'cell(x,y+1)': [],
               'x_cell(x,y)': [], 'x_cell(x-1,y)': [], 'x_cell(x,y-1)': [],
               'x_cell(x+1,y)': [], 'x_cell(x,y+1)': [],
               'line_n(x,y)': [], 'line_s(x,y)': [],
               'line_e(x,y)': [], 'line_w(x,y)': [],
               'line_n2(x,y)': [], 'line_s2(x,y)': [],
               'line_e2(x,y)': [], 'line_w2(x,y)': [],
               'x': [], 'y': []
               }

    for grid, x_grid, grid_data_row in zip(grids, x_grids, grid_data_table):
        for x in range(grid.width):
            for y in range(grid.height):
                result['cell(x,y)'].append(grid.safe_access(x, y))
                result['cell(x-1,y)'].append(grid.safe_access(x-1, y))
                result['cell(x,y-1)'].append(grid.safe_access(x, y-1))
                result['cell(x+1,y)'].append(grid.safe_access(x+1, y))
                result['cell(x,y+1)'].append(grid.safe_access(x, y+1))
                result['x_cell(x,y)'].append(x_grid.safe_access(x, y))
                result['x_cell(x-1,y)'].append(x_grid.safe_access(x-1, y))
                result['x_cell(x,y-1)'].append(x_grid.safe_access(x, y-1))
                result['x_cell(x+1,y)'].append(x_grid.safe_access(x+1, y))
                result['x_cell(x,y+1)'].append(x_grid.safe_access(x, y+1))
                result['line_s(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y),
                    grid.safe_access(x, y+1), grid.safe_access(x, y+2)]))
                result['line_e(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y),
                    grid.safe_access(x+1, y), grid.safe_access(x+2, y)]))
                result['line_n(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y),
                    grid.safe_access(x, y-1), grid.safe_access(x, y-2)]))
                result['line_w(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y),
                    grid.safe_access(x-1, y), grid.safe_access(x-2, y)]))
                result['line_s2(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y+1), grid.safe_access(x, y+2)]))
                result['line_e2(x,y)'].append(merge_pixels([
                    grid.safe_access(x+1, y), grid.safe_access(x+2, y)]))
                result['line_n2(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y-1), grid.safe_access(x, y-2)]))
                result['line_w2(x,y)'].append(merge_pixels([
                    grid.safe_access(x-1, y), grid.safe_access(x-2, y)]))

                result['x'].append(x)
                result['y'].append(y)
                for k, v in grid_data_row.items():
                    result[k].append(v)
    return pd.DataFrame(result)


def _crop(canvas: Grid, correct_df: pd.DataFrame, include_corner: bool)->Grid:
    x_values, y_values = correct_df['x'], correct_df['y']
    shapes: list[Shape] = [FilledRectangle(x, y, 1, 1, 1)
                           for x, y in zip(x_values, y_values)]

    x, y = bound_x(shapes), bound_y(shapes)
    w, h = bound_width(shapes), bound_height(shapes)
    if include_corner:
        return canvas.crop(x, y, w, h)
    else:
        return canvas.crop(x+1, y+1, w-2, h-2)
