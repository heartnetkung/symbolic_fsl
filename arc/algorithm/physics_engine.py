from abc import ABC, abstractmethod
from ..graphic import *
from ..base import *
from copy import deepcopy

MAX_SIM_STEP = 1000


class SolidSimulation:
    def __init__(self, width: int, height: int, moving_shapes: list[FilledRectangle],
                 still_shapes: list[Shape], gravity_direction: Direction)->None:

        # moving shapes are assumed to be pixel
        self.moving_shapes: list[FilledRectangle] = sort_shapes(
            moving_shapes, gravity_direction)  # type:ignore
        self.gravity_direction = gravity_direction
        canvas = draw_canvas(width, height, still_shapes)
        self.canvas_shape = Unknown(0, 0, canvas)

    def simulate(self)->Optional[list[FilledRectangle]]:
        current_shapes = self.moving_shapes
        for a in range(MAX_SIM_STEP):
            updated_shapes = self._sim_step(current_shapes)
            if updated_shapes == current_shapes:
                return updated_shapes

            current_shapes = updated_shapes
        return None

    def _sim_step(self, moving_shapes: list[FilledRectangle])->list[FilledRectangle]:
        result = []
        temp_canvas = Grid(deepcopy(self.canvas_shape.grid.data))

        for moving_shape in moving_shapes:
            offset_x, offset_y = self.gravity_direction.get_offset()
            new_shape = deepcopy(moving_shape)
            new_shape.x += offset_x
            new_shape.y += offset_y

            overlap = _is_overlap(temp_canvas, new_shape)
            out_of_bound = _is_out_of_bound(new_shape, temp_canvas)
            updated_shape = moving_shape if (overlap or out_of_bound) else new_shape
            result.append(updated_shape)
            updated_shape.draw(temp_canvas)
        return result


def _is_out_of_bound(shape: Shape, grid: Grid)->bool:
    if shape.x < 0 or shape.y < 0:
        return True
    if ((shape.x+shape.width) > grid.width) or ((shape.y+shape.height) > grid.height):
        return True
    return False


def _is_overlap(grid: Grid, pixel: FilledRectangle)->bool:
    return grid.safe_access(pixel.x, pixel.y) != NULL_COLOR
