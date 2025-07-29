from abc import ABC, abstractmethod
from ..graphic import *
from ..base import *
from copy import deepcopy

MAX_SIM_STEP = 1000


class SolidSimulation:
    def __init__(self, width: int, height: int, moving_shapes: list[Shape],
                 still_shapes: list[Shape], gravity_direction: Direction)->None:
        self.moving_shapes = sort_shapes(moving_shapes, gravity_direction)
        self.gravity_direction = gravity_direction
        canvas = draw_canvas(width, height, still_shapes)
        self.canvas_shape = Unknown(0, 0, canvas)

    def simulate(self)->Optional[list[Shape]]:
        current_shapes = self.moving_shapes
        for a in range(MAX_SIM_STEP):
            updated_shapes = self._simulate_step(current_shapes)
            if updated_shapes == current_shapes:
                return updated_shapes

            current_shapes = updated_shapes
        return None

    def _simulate_step(self, moving_shapes: list[Shape])->list[Shape]:
        result = []
        temp_canvas = Grid(deepcopy(self.canvas_shape.grid.data))
        temp_canvas_shape = Unknown(0, 0, temp_canvas)

        for moving_shape in moving_shapes:
            offset_x, offset_y = self.gravity_direction.get_offset()
            new_shape = deepcopy(moving_shape)
            new_shape.x += offset_x
            new_shape.y += offset_y

            overlap = is_overlap(temp_canvas_shape, new_shape)
            out_of_bound = _is_out_of_bound(
                new_shape, self.canvas_shape.width, self.canvas_shape.height)
            updated_shape = moving_shape if (overlap or out_of_bound) else new_shape
            result.append(updated_shape)
            updated_shape.draw(temp_canvas)
        return result


def _is_out_of_bound(shape: Shape, width: int, height: int)->bool:
    if shape.x < 0 or shape.y < 0:
        return True
    if ((shape.x+shape.width) > width) or ((shape.y+shape.height) > height):
        return True
    return False
