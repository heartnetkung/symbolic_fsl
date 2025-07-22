from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import pandas as pd
import numpy as np
from itertools import permutations
from functools import cmp_to_key


class DrawCanvas(ModelBasedArcAction[DrawCanvasTask, DrawCanvasTask]):
    def __init__(self, width_model: MLModel, height_model: MLModel,
                 params: GlobalParams, layer_model: Optional[MLModel] = None)->None:
        self.width_model = width_model
        self.height_model = height_model
        self.layer_model = layer_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: DrawCanvasTask)->Optional[ArcState]:
        assert state.out_shapes is not None
        assert state.y_bg is not None
        if self.layer_model is not None:
            out_shapes = _sort_all_shapes(state.x, state.out_shapes, self.layer_model)
        else:
            out_shapes = state.out_shapes

        df = create_df(state.x, out_shapes)
        widths = self.width_model.predict_int(df)
        heights = self.height_model.predict_int(df)
        canvases = []

        for width, height, background, shapes in zip(
                widths, heights, state.y_bg, out_shapes):
            canvas = make_grid(width, height, background)
            for shape in shapes:
                shape.draw(canvas)
            canvases.append(canvas)
        return state.update(out=canvases)

    def train_models(self, state: ArcTrainingState,
                     task: DrawCanvasTask)->list[InferenceAction]:
        assert state.y_shapes is not None
        if self.layer_model is None:
            return [self]

        assert isinstance(self.layer_model, MemorizedModel)
        df = _create_sort_df(state.y, state.y_shapes)
        if df is not None:
            models = classifier_factory(
                df, self.layer_model.result, self.params, 'draw_canvas.l')
        else:
            models = [ConstantModel(1)]
        return [DrawCanvas(self.width_model, self.height_model, self.params, model)
                for model in models]


def create_df(grids: list[Grid], all_shapes: list[list[Shape]])->pd.DataFrame:
    assert len(grids) == len(all_shapes)
    result = {'grid_width': [grid.width for grid in grids],
              'grid_height': [grid.height for grid in grids],
              'bound_width(shapes)': [bound_width(shapes) for shapes in all_shapes],
              'bound_height(shapes)': [bound_height(shapes) for shapes in all_shapes]}
    return pd.DataFrame(result)


def _create_sort_df(grids: list[Grid],
                    all_shapes: list[list[Shape]])->Optional[pd.DataFrame]:
    grids2, all_shapes2 = [], []
    for grid, shapes in zip(grids, all_shapes):
        for shape1, shape2 in permutations(shapes, 2):
            label = make_sort_label(grid, shape1, shape2)
            if label is None:
                continue

            grids2.append(grid)
            grids2.append(grid)
            all_shapes2.append([shape1, shape2])
            all_shapes2.append([shape2, shape1])

    if len(grids2) == 0:
        return None
    return generate_df(grids2, all_shapes2)


def make_sort_label(grid: Grid, a: Shape, b: Shape)->Optional[bool]:
    range_x_a, range_x_b = range(a.x, a.x+a.width), range(b.x, b.x+b.width)
    if not range_intersect(range_x_a, range_x_b):
        return None

    range_y_a, range_y_b = range(a.y, a.y+a.height), range(b.y, b.y+b.height)
    if not range_intersect(range_y_a, range_y_b):
        return None

    count1, count2 = _diff_count(grid, a, b), _diff_count(grid, b, a)
    if count1 == count2:
        return None
    return count1 > count2


def _diff_count(grid: Grid, a: Shape, b: Shape)->int:
    canvas = make_grid(grid.width, grid.height)
    a.draw(canvas)
    b.draw(canvas)

    count = 0
    for i in range(grid.height):
        for j in range(grid.width):
            canvas_cell = canvas.data[i][j]
            if canvas_cell == NULL_COLOR:
                continue
            if canvas_cell == grid.data[i][j]:
                count += 1
    return count


def _sort_all_shapes(grids: list[Grid],  all_shapes: list[list[Shape]],
                     model: MLModel)->list[list[Shape]]:
    table = _gen_sorting_table(grids, all_shapes, model)
    comp = Comparator(table)

    result = []
    for id1, (grid, shapes) in enumerate(zip(grids, all_shapes)):
        shape_with_index = [(id1, id2, shape) for id2, shape in enumerate(shapes)]
        shape_with_index.sort(key=cmp_to_key(comp.compare))
        result.append([shape for _, _, shape in shape_with_index])
    return result


def _gen_sorting_table(grids: list[Grid], all_shapes: list[list[Shape]],
                       model: MLModel)->dict[tuple[int, int, int], bool]:
    index, grid_df, shapes_df = [], [], []
    for id1, (grid, shapes) in enumerate(zip(grids, all_shapes)):
        for i, j in permutations(range(len(shapes)), 2):
            shape1, shape2 = shapes[i], shapes[j]
            label = make_sort_label(grid, shape1, shape2)
            if label is None:
                continue

            index.append((id1, i, j))
            grid_df.append(grid)
            shapes_df.append([shape1, shape2])

    if len(grid_df) == 0:
        return {}

    df = generate_df(grid_df, shapes_df)
    prediction = model.predict_bool(df)
    return {key: pred for pred, key in zip(prediction, index)}


class Comparator:
    def __init__(self, table: dict[tuple[int, int, int], bool])->None:
        self.table = table

    def compare(self, a: tuple[int, int, Shape], b: tuple[int, int, Shape])->int:
        key1, key2 = (a[0], a[1], b[1]), (a[0], b[1], a[1])
        value1, value2 = self.table.get(key1, None), self.table.get(key2, None)
        if value1 == True or value2 == False:
            return 1
        elif value1 == False or value2 == True:
            return -1
        else:
            return 1  # lookup fails
