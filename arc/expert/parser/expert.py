from ...base import *
from ...graphic import *
from ...manager.task import *
from .independent_parse import ParseMode, IndependentParse
from .persistent_parse import PersistentParseMode, PersistentParse
from itertools import product
from ...algorithm.find_background import find_backgrounds


class ParseGridExpert(Expert[ArcTrainingState, ParseGridTask]):
    def solve_problem(self, state: ArcTrainingState, task: ParseGridTask)->list[Action]:
        result = []
        backgrounds = find_backgrounds(state)

        # independent parse
        for x_mode, y_mode, (x_model, y_model), unknown_bg in product(
                ParseMode, ParseMode, backgrounds, BOOLS):
            if unknown_bg:  # proximity + unknown_bg == crop
                if x_mode in (ParseMode.proximity_diag, ParseMode.proximity_normal):
                    continue
                if y_mode in (ParseMode.proximity_diag, ParseMode.proximity_normal):
                    continue
            if x_mode == ParseMode.partition:
                if not _have_partitions(state.x):
                    continue
            if y_mode == ParseMode.partition:
                if not _have_partitions(state.y):
                    continue
            result.append(IndependentParse(
                x_mode, y_mode, x_model, y_model, unknown_bg))

        # persistent parse
        for x_mode, y_mode, (x_model, y_model) in product(
                PersistentParseMode, PersistentParseMode, backgrounds):
            result.append(PersistentParse(x_mode, y_mode, x_model, y_model))

        return result


def _have_partitions(grids: list[Grid])->bool:
    len_grids = len(grids)
    for sample_id, grid in enumerate(grids):
        rows, cols, row_colors, col_colors = find_separators(grid)
        if len(row_colors) == 0 and len(col_colors) == 0:
            return False
    return True
