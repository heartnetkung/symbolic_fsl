from .apply_partitionless_logic import *
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from itertools import combinations
from ..util import *
from ...attention import list_shape_representations


class ApplyPartitionlessLogicExpert(Expert[ArcTrainingState, PartitionlessLogicTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: PartitionlessLogicTask)->list[Action]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None

        result: list[Action] = [ApplyPartitionlessLogic(
            PartitionlessLogicParam.skip, self.params)]
        is_horizontal = _is_horizontal(state.x, state.y)
        if is_horizontal is None:
            return result
        if not is_single_fullsize_shape(state.out_shapes, state.x):
            return result

        splitted_shapes = split_shapes_equally(state.out_shapes, is_horizontal)
        if splitted_shapes is None:
            return result

        for color in _find_common_colors(splitted_shapes):
            for type in LogicType:
                candidate = ApplyPartitionlessLogic(
                    PartitionlessLogicParam.normal, self.params,
                    color, type, is_horizontal)
                produced_shapes = candidate.apply(state)
                if produced_shapes is None:
                    continue
                if not _check_result(produced_shapes, state.y_shapes):
                    continue

                result.append(candidate)
        return result


def _find_common_colors(all_shapes: list[list[Shape]])->set[int]:
    all_colors = []
    for shapes in all_shapes:
        all_colors += [shape._grid.list_colors() for shape in shapes]
    return set.intersection(*all_colors)


def _check_result(all_a_shapes: list[list[Shape]],
                  all_b_shapes: list[list[Shape]])->bool:
    assert len(all_a_shapes) == len(all_b_shapes)

    for a_shapes, b_shapes in zip(all_a_shapes, all_b_shapes):
        a_values = [list_shape_representations(a)['colorless_transformed_shape']
                    for a in a_shapes]
        b_values = [list_shape_representations(b)['colorless_transformed_shape']
                    for b in b_shapes]
        if a_values != b_values:
            return False
    return True


def _is_horizontal(x_grids: list[Grid], y_grids: list[Grid])->Optional[bool]:
    result = set()
    for x_grid, y_grid in zip(x_grids, y_grids):
        width_equal = x_grid.width == y_grid.width
        height_equal = x_grid.height == y_grid.height
        if width_equal and height_equal:
            return None
        if (not width_equal) and (not height_equal):
            return None
        result.add(width_equal)

    if len(result) != 1:
        return None
    return result.pop()
