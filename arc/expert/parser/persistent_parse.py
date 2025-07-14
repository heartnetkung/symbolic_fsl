from ...base import *
from ...graphic import *
from ...manager.task import ParseGridTask
from enum import Enum
from copy import deepcopy
from ...algorithm.find_background import make_background_df
from ...ml import *


class PersistentParseMode(Enum):
    proximity_diag = 0
    # proximity_normal = 1
    color_proximity_diag = 2
    # color_proximity_normal = 3


class PersistentParse(ModelFreeArcAction):
    def __init__(self, x_mode: PersistentParseMode, y_mode: PersistentParseMode,
                 x_bg_model: MLModel, y_bg_model: MLModel)->None:
        self.x_mode = x_mode
        self.y_mode = y_mode
        self.x_bg_model = x_bg_model
        self.y_bg_model = y_bg_model
        super().__init__()

    def perform(self, state: ArcState, is_training: bool)->Optional[ArcState]:
        df = make_background_df(state)
        x_bg = self.x_bg_model.predict_int(df)
        y_bg = self.y_bg_model.predict_int(df)

        x_shapes = self._parse(state.x, self.x_mode, x_bg)
        if not is_training:
            return state.update(out_shapes=x_shapes, x_shapes=x_shapes,
                                x_bg=x_bg, y_bg=y_bg)

        assert isinstance(state, ArcTrainingState)
        y_shapes = self._parse(state.y, self.y_mode, y_bg)
        return state.update(out_shapes=x_shapes, x_shapes=x_shapes,
                            x_bg=x_bg, y_bg=y_bg, y_shapes=y_shapes)

    def _parse(self, grids: list[Grid], mode: PersistentParseMode,
               backgrounds: list[int])->list[list[Shape]]:
        grids2 = [grid.replace_color(bg, NULL_COLOR)
                  for grid, bg in zip(grids, backgrounds)]
        if mode == PersistentParseMode.proximity_diag:
            return [list_sparse_objects(grid, True)for grid in grids2]
        return [list_objects(grid, True) for grid in grids2]

    def _parse_diff(
            self, grids: list[Grid], mode: PersistentParseMode, backgrounds: list[int],
            all_diff_shapes: list[list[Shape]])->list[list[Shape]]:
        result = []
        for grid, diff_shapes, bg in zip(grids, all_diff_shapes, backgrounds):
            grid = grid.replace_color(bg, NULL_COLOR)
            new_result = []

            for diff_shape in diff_shapes:
                grid, new_shape = self._diff(grid, diff_shape)
                if new_shape is not None:
                    new_result.append(new_shape)

            if mode == PersistentParseMode.proximity_diag:
                new_result += list_sparse_objects(grid, True)
            else:
                new_result += list_objects(grid, True)
            result.append(new_result)
        return result

    def _diff(self, grid: Grid, diff_shape: Shape)->tuple[Grid, Optional[Shape]]:
        for i in range(diff_shape.height):
            for j in range(diff_shape.width):
                if diff_shape._grid.data[i][j] == NULL_COLOR:
                    continue
                i2, j2 = i+diff_shape.y, j+diff_shape.x
                if grid.safe_access(j2, i2) in (NULL_COLOR, MISSING_VALUE):
                    return grid, None

        grid2 = deepcopy(grid)
        new_shape_grid = make_grid(diff_shape.width, diff_shape.height)
        for i in range(diff_shape.height):
            for j in range(diff_shape.width):
                if diff_shape._grid.data[i][j] == NULL_COLOR:
                    continue

                i2, j2 = i+diff_shape.y, j+diff_shape.x
                new_shape_grid.data[i][j] = grid2.data[i2][j2]
                grid2.data[i2][j2] = NULL_COLOR
        return grid2, from_grid(diff_shape.x, diff_shape.y, new_shape_grid)
