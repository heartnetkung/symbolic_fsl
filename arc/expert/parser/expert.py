from ...base import *
from ...graphic import *
from ...manager.task import *
from .independent_parse import ParseMode, IndependentParse
from itertools import product
from ...algorithm.find_background import find_backgrounds
from collections import Counter


class ParseGridExpert(Expert[ArcTrainingState, ParseGridTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState, task: ParseGridTask)->list[Action]:
        result = []
        backgrounds = find_backgrounds(state)
        x_partition_color = _get_partition_color(state.x)
        y_partition_color = _get_partition_color(state.y)

        # independent parse
        for x_mode, y_mode, (x_model, y_model), unknown_bg in product(
                self.params.parser_x_modes, self.params.parser_y_modes,
                backgrounds, BOOLS):
            if unknown_bg:  # proximity + unknown_bg == crop
                if x_mode in (ParseMode.proximity_diag, ParseMode.proximity_normal):
                    continue
                if y_mode in (ParseMode.proximity_diag, ParseMode.proximity_normal):
                    continue
            if x_mode == ParseMode.partition:
                if x_partition_color is None:
                    continue
            if y_mode == ParseMode.partition:
                if y_partition_color is None:
                    continue

            result.append(IndependentParse(
                x_mode, y_mode, x_model, y_model, unknown_bg,
                x_partition_color, y_partition_color))
        return result


def _have_partitions(grids: list[Grid])->bool:
    len_grids = len(grids)
    for sample_id, grid in enumerate(grids):
        rows, cols, row_colors, col_colors = find_separators(grid)
        if len(row_colors) == 0 and len(col_colors) == 0:
            return False
    return True


def _get_partition_color(grids: list[Grid])->Optional[int]:
    counter = Counter()
    for sample_id, grid in enumerate(grids):
        rows, cols, row_colors, col_colors = find_separators(grid)
        total_colors = set(row_colors+col_colors)
        if len(total_colors) == 0:
            return None

        counter.update(total_colors)

    color, count = counter.most_common(1)[0]
    return color if count == len(grids) else NULL_COLOR
