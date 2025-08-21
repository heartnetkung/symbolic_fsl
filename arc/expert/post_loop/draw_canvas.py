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
                 params: GlobalParams, layer_model: Optional[MLModel] = None,
                 is_all_equal: bool = False)->None:
        self.width_model = width_model
        self.height_model = height_model
        self.layer_model = layer_model
        self.params = params
        self.is_all_equal = is_all_equal
        super().__init__()

    def perform(self, state: ArcState, task: DrawCanvasTask)->Optional[ArcState]:
        assert state.out_shapes is not None
        assert state.y_bg is not None
        if self.layer_model is not None:
            out_shapes = _sort_all_shapes(
                state.x, state.out_shapes, self.layer_model, self.is_all_equal)
        else:
            out_shapes = state.out_shapes

        df = create_df(state.x, out_shapes)
        widths = self.width_model.predict_int(df)
        heights = self.height_model.predict_int(df)
        canvases = []

        for width, height, background, shapes in zip(
                widths, heights, state.y_bg, out_shapes):
            if (width <= 0) or (height <= 0):
                return None

            new_canvas = draw_canvas(width, height, shapes, background)
            if _has_invalid_color(new_canvas):
                return None

            canvases.append(new_canvas)
        return state.update(out=canvases)

    def train_models(self, state: ArcTrainingState,
                     task: DrawCanvasTask)->list[InferenceAction]:
        assert state.y_shapes is not None
        if self.layer_model is None:
            return [self]

        assert isinstance(self.layer_model, MemorizedModel)
        is_all_equal = state.x_shapes == state.y_shapes
        df = _create_sort_df(state.y, state.y_shapes, state.x, is_all_equal)
        if df is not None:
            models = make_classifier(
                df, self.layer_model.result, self.params, 'draw_canvas.l')
        else:
            models = [ConstantModel(1)]
        return [DrawCanvas(self.width_model, self.height_model, self.params,
                           model, is_all_equal) for model in models]


def create_df(grids: list[Grid], all_shapes: list[list[Shape]])->pd.DataFrame:
    assert len(grids) == len(all_shapes)
    df = generate_df(grids)
    df['bound_width(shapes)'] = [bound_width(shapes) for shapes in all_shapes]
    df['bound_height(shapes)'] = [bound_height(shapes) for shapes in all_shapes]
    return df


def _has_invalid_color(grid: Grid)->bool:
    for i in range(grid.height):
        for j in range(grid.width):
            if not valid_color(grid.data[i][j]):
                return True
    return False


def _create_sort_df(y_grids: list[Grid], all_y_shapes: list[list[Shape]],
                    x_grids: list[Grid], is_all_equal: bool)->Optional[pd.DataFrame]:
    grids2, all_shapes2, x_labels = [], [], []
    for y_grid, y_shapes, x_grid in zip(y_grids, all_y_shapes, x_grids):
        for shape1, shape2 in permutations(y_shapes, 2):
            label = make_sort_label(y_grid, shape1, shape2)
            if label is None:
                continue

            if is_all_equal:
                x_label = make_sort_label(x_grid, shape1, shape2)
                if x_label is not None:
                    x_labels.append(not x_label)
                    x_labels.append(x_label)

            grids2.append(y_grid)
            grids2.append(y_grid)
            all_shapes2.append([shape1, shape2])
            all_shapes2.append([shape2, shape1])

    if len(grids2) == 0:
        return None
    result = generate_df(grids2, all_shapes2)
    if len(x_labels) == len(result):
        result['x_label'] = x_labels
    return result


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
                     model: MLModel, is_all_equal: bool)->list[list[Shape]]:
    table = _gen_sorting_table(grids, all_shapes, model, is_all_equal)
    comp = Comparator(table)

    result = []
    for id1, (grid, shapes) in enumerate(zip(grids, all_shapes)):
        shape_with_index = [(id1, id2, shape) for id2, shape in enumerate(shapes)]
        shape_with_index.sort(key=cmp_to_key(comp.compare))
        result.append([shape for _, _, shape in shape_with_index])
    return result


def _gen_sorting_table(
        grids: list[Grid], all_shapes: list[list[Shape]],
        model: MLModel, is_all_equal: bool)->dict[tuple[int, int, int], bool]:
    index, grid_df, shapes_df, x_labels = [], [], [], []
    for id1, (grid, shapes) in enumerate(zip(grids, all_shapes)):
        for i, j in permutations(range(len(shapes)), 2):
            shape1, shape2 = shapes[i], shapes[j]
            label = make_sort_label(grid, shape1, shape2)
            if label is None:
                continue

            if is_all_equal:
                x_labels.append(0 if label else 1)

            index.append((id1, i, j))
            grid_df.append(grid)
            shapes_df.append([shape1, shape2])

    if len(grid_df) == 0:
        return {}

    df = generate_df(grid_df, shapes_df)
    if len(x_labels) == len(df):
        df['x_label'] = x_labels
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
