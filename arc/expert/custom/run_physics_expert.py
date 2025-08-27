from .run_physics import RunPhysics, RunPhysicsParam
from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from ..util import *


class RunPhysicsExpert(Expert[ArcTrainingState, PhysicsTask]):
    def __init__(self, params: GlobalParams)->None:
        self.params = params

    def solve_problem(self, state: ArcTrainingState, task: PhysicsTask)->list[Action]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None

        result: list[Action] = [RunPhysics(RunPhysicsParam.skip, set(), self.params)]
        if not _check_mass(state.out_shapes, state.y_shapes, state.x, state.y):
            return result

        still_colors = _get_still_colors(state.out_shapes, state.y_shapes)
        if still_colors is None:
            return result

        for param in RunPhysicsParam:
            if param == RunPhysicsParam.skip:
                continue

            new_action = RunPhysics(param, still_colors, self.params)
            out_state = new_action.perform(state, task)
            if out_state is None:
                continue
            if not isinstance(out_state, ArcTrainingState):
                continue
            if (out_state.out_shapes is None) or (out_state.y_shapes is None):
                continue
            if not _check_result(out_state.out_shapes, out_state.y_shapes):
                continue

            result.append(new_action)
        return result


def _check_mass(all_x_shapes: list[list[Shape]], all_y_shapes: list[list[Shape]],
                x_grids: list[Grid], y_grids: list[Grid])->bool:
    for x_grid, y_grid, x_shapes, y_shapes in zip(
            x_grids, y_grids, all_x_shapes, all_y_shapes):
        if x_grid.width != y_grid.width:
            return False
        if x_grid.height != y_grid.height:
            return False

        area = x_grid.width*x_grid.height
        x_mass = sum([shape.mass for shape in x_shapes])
        y_mass = sum([shape.mass for shape in y_shapes])
        if x_mass != y_mass:
            return False
        if x_mass == area:
            return False
    return True


def _get_still_colors(all_x_shapes: list[list[Shape]],
                      all_y_shapes: list[list[Shape]])->Optional[set[int]]:

    all_common_colors = []
    for x_shapes, y_shapes in zip(all_x_shapes, all_y_shapes):
        x_groups, y_groups = _group_colors(x_shapes), _group_colors(y_shapes)
        if (x_groups is None) or (y_groups is None):
            return None

        common_colors = set()
        for color, x_group in x_groups.items():
            if x_group == y_groups.get(color, None):
                common_colors.add(color)
        all_common_colors.append(common_colors)
    return set.intersection(*all_common_colors)


def _group_colors(shapes: list[Shape])->Optional[dict[int, set[Shape]]]:
    result = {}
    for shape in shapes:
        color = shape.single_color
        if color == NULL_COLOR:
            return None

        group = result.get(color, set())
        group.add(shape)
        result[color] = group
    return result


def _check_result(all_x_shapes: list[list[Shape]],
                  all_y_shapes: list[list[Shape]])->bool:
    for x_shapes, y_shapes in zip(all_x_shapes, all_y_shapes):
        colorless_x = {colorize(shape, 1) for shape in x_shapes}
        colorless_y = {colorize(shape, 1) for shape in y_shapes}
        if colorless_x != colorless_y:
            return False
    return True
