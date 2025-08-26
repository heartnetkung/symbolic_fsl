from ...base import *
from ...graphic import *
from ...manager.task import *
from .independent_parse import ParseMode, IndependentParse
from itertools import product
from ...algorithm.find_background import find_backgrounds, make_background_df
from collections import Counter


class ParseGridExpert(Expert[ArcTrainingState, ParseGridTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState, task: ParseGridTask)->list[Action]:
        result = []
        backgrounds = dict.fromkeys(find_backgrounds(state))
        x_partition_colors = _cache_partition_color(state.x)
        y_partition_colors = _cache_partition_color(state.y)
        df = make_background_df(state)
        y_constant_size = _get_constant_y_size(state.y)

        # independent parse
        for x_mode, y_mode, (x_model, y_model), unknown_bg in product(
                self.params.parser_x_modes, self.params.parser_y_modes,
                backgrounds, BOOLS):
            x_partition_color, y_partition_color = NULL_COLOR, NULL_COLOR

            if unknown_bg:  # proximity + unknown_bg == crop
                if x_mode in (ParseMode.proximity_diag, ParseMode.proximity_normal):
                    continue
                if y_mode in (ParseMode.proximity_diag, ParseMode.proximity_normal):
                    continue
            if y_mode == ParseMode.partition_by_size:
                continue
            if x_mode == ParseMode.partition_by_size:
                if y_mode != ParseMode.crop:
                    continue
                if unknown_bg:
                    continue
            if x_mode == ParseMode.partition:
                if x_partition_colors is None:
                    continue
                x_bg = x_model.predict_int(df)
                x_partition_color = _get_partition_color(x_partition_colors, x_bg)
            if y_mode == ParseMode.partition:
                if y_partition_colors is None:
                    continue
                y_bg = y_model.predict_int(df)
                y_partition_color = _get_partition_color(y_partition_colors, y_bg)

            result.append(IndependentParse(
                x_mode, y_mode, x_model, y_model, unknown_bg,
                x_partition_color, y_partition_color, y_constant_size))
        return result


def _cache_partition_color(grids: list[Grid])->Optional[set[int]]:
    counter = Counter()
    for sample_id, grid in enumerate(grids):
        rows, cols, row_colors, col_colors = find_separators(grid)
        total_colors = set(row_colors+col_colors) - {0}
        if len(total_colors) == 0:
            return None

        counter.update(total_colors)

    len_grids = len(grids)
    return {color for color, count in counter.most_common() if count == len_grids}


def _get_partition_color(possible_colors: set[int], backgrounds: list[int])->int:
    to_subtract = set(backgrounds)
    if len(to_subtract) != 1:  # only subtract if the background is constant
        to_subtract = set()

    result = possible_colors - to_subtract
    if len(result) != 1:
        return NULL_COLOR
    return result.pop()


def _get_constant_y_size(y_grids: list[Grid])->Optional[tuple[int, int]]:
    first_size = (y_grids[0].width, y_grids[0].height)
    for i in range(1, len(y_grids)):
        if (y_grids[i].width, y_grids[i].height) != first_size:
            return None
    return first_size
