from ...base import *
from ...graphic import *
from ...ml import *
from copy import deepcopy
from ..util import *
from ...manager.task import PhysicsTask
from enum import Enum
from ...algorithm.physics_engine import SolidSimulation


class RunPhysicsParam(Enum):
    # the numbers are intentional
    skip = -1
    solid_north = 0
    solid_east = 2
    solid_south = 4
    solid_west = 6

    def get_gravity_direction(self)->Direction:
        try:
            return Direction(self.value % 10)
        except ValueError:
            raise Exception('unsupported value')


class RunPhysics(ModelFreeArcAction[PhysicsTask]):
    def __init__(self, param: RunPhysicsParam, still_colors: set[int],
                 params: GlobalParams)->None:
        self.param = param
        self.still_colors = still_colors
        self.params = params

    def perform(self, state: ArcState, task: PhysicsTask)->Optional[ArcState]:
        assert state.out_shapes is not None
        if self.param == RunPhysicsParam.skip:
            return state

        dir_ = self.param.get_gravity_direction()
        new_out_shapes = []
        for x_shapes, grid in zip(state.out_shapes, state.x):
            blob = _extract_shapes(x_shapes, self.still_colors, grid.width, grid.height)
            if blob is None:
                return None

            still_shapes, moving_shapes = blob
            simulator = SolidSimulation(
                grid.width, grid.height, moving_shapes, still_shapes, dir_)
            updated_shapes = simulator.simulate()
            if updated_shapes is None:
                return None

            new_out_shapes.append(still_shapes+updated_shapes)

        if not isinstance(state, ArcTrainingState):
            return state.update(out_shapes=new_out_shapes,
                                reparse_count=self.params.max_reparse)

        assert state.y_shapes is not None
        new_y_shapes = []
        for y_shapes, grid in zip(state.y_shapes, state.y):
            blob = _extract_shapes(y_shapes, self.still_colors, grid.width, grid.height)
            if blob is None:
                return None

            new_y_shapes.append(blob[0]+blob[1])
        return state.update(out_shapes=new_out_shapes, y_shapes=new_y_shapes,
                            reparse_count=self.params.max_reparse)


def _extract_shapes(shapes: list[Shape], still_colors: set[int], width: int,
                    height: int)->Optional[tuple[list[Shape], list[FilledRectangle]]]:
    still_shapes, moving_shapes = [], []
    for shape in shapes:
        color = shape.get_single_color()
        if color == NULL_COLOR:
            return None

        if color in still_colors:
            still_shapes.append(shape)
        else:
            moving_shapes.append(shape)
    return still_shapes, _parse_to_pixels(moving_shapes, width, height)


def _parse_to_pixels(shapes: list[Shape], w: int, h: int)->list[FilledRectangle]:
    canvas = draw_canvas(w, h, shapes)
    result = []
    for i, row in enumerate(canvas.data):
        for j, cell in enumerate(row):
            if cell != NULL_COLOR:
                result.append(FilledRectangle(j, i, 1, 1, cell))
    return result
