from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import pandas as pd
from ..util import *
from copy import deepcopy


class Crop(ModelBasedArcAction[CropTask, CropTask]):
    def __init__(self, corner_finder: MLModel, params: GlobalParams,
                 include_corner: bool = True)->None:
        self.corner_finder = corner_finder
        self.include_corner = include_corner
        self.params = params

    def perform(self, state: ArcState, task: CropTask)->Optional[ArcState]:
        assert state.out_shapes != None

        new_out_shapes = []
        for canvas in self._get_canvases(state, task):
            df = _make_df([canvas])
            if df is None:
                return None

            corners = self.corner_finder.predict_bool(df)
            if np.sum(corners) != 4:
                return None

            correct_df = df[corners]
            assert isinstance(correct_df, pd.DataFrame)
            new_grid = _crop(canvas, correct_df, self.include_corner)
            if new_grid is None:
                return None

            new_out_shapes.append([Unknown(0, 0, new_grid)])
        return state.update(out_shapes=new_out_shapes)

    def train_models(self, state: ArcTrainingState,
                     task: CropTask)->list[InferenceAction]:
        # check include_corner
        assert state.out_shapes is not None
        assert isinstance(self.corner_finder, StepMemoryModel)

        canvases = self._get_canvases(state, task)
        df = _make_df(canvases)
        if df is None:
            return []

        models = regressor_factory(df, self.corner_finder.result, self.params, 'crop')
        return [Crop(model, self.params) for model in models]

    def _get_canvases(self, state: ArcState, task: CropTask)->list[Grid]:
        if task.crop_from_out_shapes:
            assert state.out_shapes is not None
            return [draw_canvas(grid.width, grid.height, shapes)
                    for grid, shapes in zip(state.x, state.out_shapes)]
        return state.x


def _make_df(grids: list[Grid])->Optional[pd.DataFrame]:
    if len(grids) == 0:
        return None

    grid_data_table = generate_df(grids).to_dict('records')
    result = {col: [] for col in grid_data_table[0]}
    result |= {'cell(x,y)': [], 'cell(x-1,y)': [], 'cell(x,y-1)': [],
               'cell(x+1,y)': [], 'cell(x,y+1)': [], 'x': [], 'y': []}

    for grid, grid_data_row in zip(grids, grid_data_table):
        for x in range(grid.width):
            for y in range(grid.height):
                result['cell(x,y)'].append(grid.safe_access(x, y))
                result['cell(x-1,y)'].append(grid.safe_access(x-1, y))
                result['cell(x,y-1)'].append(grid.safe_access(x, y-1))
                result['cell(x+1,y)'].append(grid.safe_access(x+1, y))
                result['cell(x,y+1)'].append(grid.safe_access(x, y+1))
                result['x'].append(x)
                result['y'].append(y)
                for k, v in grid_data_row.items():
                    result[k].append(v)
    return pd.DataFrame(result)


def _crop(canvas: Grid, correct_df: pd.DataFrame, include_corner: bool)->Optional[Grid]:
    x_values, y_values = correct_df['x'], correct_df['y']
    shapes: list[Shape] = [FilledRectangle(x, y, 1, 1, 1)
                           for x, y in zip(x_values, y_values)]

    if include_corner:
        x, y = bound_x(shapes), bound_y(shapes)
        w, h = bound_width(shapes), bound_height(shapes)
        return canvas.crop(x, y, w, h)

    first_shape = shapes[0]
    second_shapes = [shape for shape in shapes[1:]
                     if (shape.x != first_shape.x) and (shape.y != first_shape.y)]
    if len(second_shapes) != 1:
        return None
    bound = find_inner_bound(first_shape, second_shapes[0])
    return canvas.crop(bound.x, bound.y, bound.width, bound.height)
