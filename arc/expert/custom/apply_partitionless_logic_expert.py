from .apply_partitionless_logic import *
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from itertools import combinations
from ..util import *
from ...attention import list_shape_representations
from .apply_partitionless_union import ApplyPartitionlessUnion
from itertools import permutations

PERMUTATION_LIMIT = 6


class ApplyPartitionlessLogicExpert(Expert[ArcTrainingState, PartitionlessLogicTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState,
                      task: PartitionlessLogicTask)->list[Action]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None

        result: list[Action] = [ApplyPartitionlessLogic(
            PartitionlessLogicParam.skip, self.params)]
        partition = _find_partition(state.x, state.y)
        if partition is None:
            return result
        if not is_single_fullsize_shape(state.out_shapes, state.x):
            return result

        row_count, col_count = partition
        splitted_shapes = split_shapes_equally(state.out_shapes, row_count, col_count)
        if splitted_shapes is None:
            return result

        for color in _find_common_colors(splitted_shapes):
            for type in LogicType:
                candidate = ApplyPartitionlessLogic(
                    PartitionlessLogicParam.normal, self.params,
                    color, type, row_count, col_count)
                produced_shapes = candidate.apply(state)
                if produced_shapes is None:
                    continue
                if not _check_result(produced_shapes, state.y_shapes):
                    continue

                result.append(candidate)

        if row_count*col_count <= PERMUTATION_LIMIT:
            for perm in permutations(range(row_count*col_count)):
                candidate = ApplyPartitionlessUnion(
                    list(perm), row_count, col_count, self.params)
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


def _find_partition(x_grids: list[Grid],
                    y_grids: list[Grid])->Optional[tuple[int, int]]:
    n_rows, n_cols = set(), set()
    for x_grid, y_grid in zip(x_grids, y_grids):
        if x_grid.width < y_grid.width:
            return None
        if x_grid.height < y_grid.height:
            return None
        if (x_grid.width % y_grid.width) != 0:
            return None
        if (x_grid.height % y_grid.height) != 0:
            return None

        n_rows.add(round(x_grid.height / y_grid.height))
        n_cols.add(round(x_grid.width / y_grid.width))
    if (len(n_rows) != 1) or (len(n_cols) != 1):
        return None

    result = (n_rows.pop(), n_cols.pop())
    if result == (1, 1):
        return None
    return result
