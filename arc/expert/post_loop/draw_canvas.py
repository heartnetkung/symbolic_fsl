from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import pandas as pd
from itertools import permutations
from functools import cmp_to_key


class DrawCanvas(UniversalArcAction):
    def __init__(self, width_model: MLModel, height_model: MLModel,
                 layer_model: Optional[MLModel] = None)->None:
        self.width_model = width_model
        self.height_model = height_model
        self.layer_model = layer_model
        super().__init__()

    def perform(self, state: ArcState, is_training: bool)->Optional[ArcState]:
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


def create_df(grids: list[Grid], all_shapes: list[list[Shape]])->pd.DataFrame:
    assert len(grids) == len(all_shapes)
    result = {'grid_width': [grid.width for grid in grids],
              'grid_height': [grid.height for grid in grids],
              'bound_width(shapes)': [bound_width(shapes) for shapes in all_shapes],
              'bound_height(shapes)': [bound_height(shapes) for shapes in all_shapes]}
    return pd.DataFrame(result)


def create_sort_df(grids: list[Grid],
                   all_shapes: list[list[Shape]])->pd.DataFrame:
    grids2, all_shapes2 = [], []
    for grid, shapes in zip(grids, all_shapes):
        for shape1, shape2 in permutations(shapes, 2):
            grids2.append(grid)
            all_shapes2.append([shape1, shape2])
    return generate_df(grids2, all_shapes2)


def _sort_shapes(grid: Grid, shapes: list[Shape], model: MLModel)->list[Shape]:
    def _compare(a: Shape, b: Shape)->int:
        df = generate_df([grid], [[a, b]])
        result = model.predict_bool(df)[0]
        print(a, b, result)
        return 1 if result else -1
    return sorted(shapes, key=cmp_to_key(_compare))


def _sort_all_shapes(grids: list[Grid],  all_shapes: list[list[Shape]],
                     model: MLModel)->list[list[Shape]]:
    return [_sort_shapes(grid, shapes, model)
            for grid, shapes in zip(grids, all_shapes)]
