from ...base import *
from ...graphic import *
from enum import Enum
from ...manager.task import ParseGridTask
from ...algorithm.find_background import make_background_df
from ...ml import *
from ...constant import *
from ..util import filter_overwhelming_shapes


MASS_THRESHOLD = 0.4


class IndependentParse(ModelFreeArcAction[ParseGridTask]):
    def __init__(self, x_mode: ParseMode, y_mode: ParseMode, x_bg_model: MLModel,
                 y_bg_model: MLModel, unknown_background: bool,
                 x_partition_color: Optional[int] = None,
                 y_partition_color: Optional[int] = None)->None:
        self.x_mode = x_mode
        self.y_mode = y_mode
        self.x_bg_model = x_bg_model
        self.y_bg_model = y_bg_model
        self.unknown_background = unknown_background
        self.x_partition_color = x_partition_color
        self.y_partition_color = y_partition_color
        super().__init__()

    def perform(self, state: ArcState, task: ParseGridTask)->Optional[ArcState]:
        df = make_background_df(state)
        x_bg = self.x_bg_model.predict_int(df)
        y_bg = self.y_bg_model.predict_int(df)

        x_shapes = self._perform(state.x, self.x_mode, x_bg, self.x_partition_color)
        if x_shapes is None:
            return None
        x_shapes = filter_overwhelming_shapes(x_shapes)
        if not isinstance(state, ArcTrainingState):
            return state.update(out_shapes=x_shapes, x_shapes=x_shapes,
                                x_bg=x_bg, y_bg=y_bg)

        y_shapes = self._perform(state.y, self.y_mode, y_bg, self.y_partition_color)
        if y_shapes is None:
            return None
        y_shapes = filter_overwhelming_shapes(y_shapes)
        return state.update(out_shapes=x_shapes, x_shapes=x_shapes,
                            x_bg=x_bg, y_bg=y_bg, y_shapes=y_shapes)

    def _perform(self, grids: list[Grid], mode: ParseMode,
                 backgrounds: list[int],
                 partition_color: Optional[int])->Optional[list[list[Shape]]]:
        grids2 = [grid.replace_color(bg, NULL_COLOR)
                  for grid, bg in zip(grids, backgrounds)]
        unknown_grids = grids if self.unknown_background else grids2

        if mode == ParseMode.crop:
            return [[Unknown(0, 0, grid)] for grid in unknown_grids]
        if mode == ParseMode.partition:
            assert partition_color is not None
            return _make_partition_shapes(unknown_grids, backgrounds, partition_color)
        if mode == ParseMode.proximity_diag:
            return [list_sparse_objects(grid, True)for grid in grids2]
        if mode == ParseMode.proximity_normal:
            return [list_sparse_objects(grid, False) for grid in grids2]
        if mode == ParseMode.color_proximity_diag:
            return [list_objects(grid, True) for grid in grids2]
        if mode == ParseMode.color_proximity_normal:
            return [list_objects(grid, False) for grid in grids2]
        raise Exception('unsupported parse mode')


def _make_partition_shapes(
        grids: list[Grid], backgrounds: list[int],
        partition_color: int)->Optional[list[list[Shape]]]:
    result = []
    for grid, bg in zip(grids, backgrounds):
        rows, cols, row_colors, col_colors = find_separators(grid, partition_color)
        if len(row_colors)+len(col_colors) == 0:
            return None

        partitions = partition(grid, rows, cols, row_colors, col_colors)
        separator = _resolve_separator_color(partition_color, row_colors+col_colors)
        if separator in (NULL_COLOR, bg):
            return None

        grid_shapes = list_objects(grid.keep_color(separator))
        if len(grid_shapes) != 1:
            return None
        if (grid_shapes[0].mass / (grid.width*grid.height)) > MASS_THRESHOLD:
            return None

        result.append(partitions+grid_shapes)
    return result


def _resolve_separator_color(consistent_color: int, local_colors: list[int])->int:
    if consistent_color != NULL_COLOR:
        return consistent_color
    color_set = set(local_colors)
    if len(color_set) != 1:
        return NULL_COLOR
    return color_set.pop()
