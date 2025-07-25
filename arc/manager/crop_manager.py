from ..base import *
from ..graphic import *
from .task import CropTask


class CropManager(Manager[ArcTrainingState]):
    def __init__(self)->None:
        self.is_size_correct_cache: Optional[bool] = None
        self.is_bg_correct_cache: Optional[bool] = None
        self.is_initially_croppable_cache: Optional[bool] = None

    def decide(self, state: ArcTrainingState)->list[
            tuple[Task[ArcTrainingState], ArcTrainingState]]:
        assert state.y_bg is not None
        assert state.out_shapes is not None

        if not self.is_size_correct(state):
            return []

        if self.is_initially_croppable(state):
            return [(CropTask(False), state)]

        canvases = [draw_canvas(grid.width, grid.height, shapes, bg)
                    for grid, shapes, bg in zip(state.x, state.out_shapes, state.y_bg)]
        if not _is_crop(canvases, state.y):
            return []
        return [(CropTask(True), state)]

    def is_bg_correct(self, state: ArcTrainingState)->bool:
        if self.is_bg_correct_cache is None:
            assert state.x_bg is not None
            assert state.y_bg is not None
            self.is_bg_correct_cache = (state.x_bg == state.y_bg)
        return self.is_bg_correct_cache

    def is_size_correct(self, state: ArcTrainingState)->bool:
        if self.is_size_correct_cache is None:
            self.is_size_correct_cache = _check_size(state.x, state.y)
        return self.is_size_correct_cache

    def is_initially_croppable(self, state: ArcTrainingState)->bool:
        if self.is_initially_croppable_cache is None:
            self.is_initially_croppable_cache = _is_crop(state.x, state.y)
        return self.is_initially_croppable_cache


def _check_size(x_grids: list[Grid], y_grids: list[Grid])->bool:
    x_size = [(grid.width, grid.height) for grid in x_grids]
    y_size = [(grid.width, grid.height) for grid in y_grids]
    if x_size == y_size:
        return False

    for (x_width, x_height), (y_width, y_height) in zip(x_size, y_size):
        if (x_width < y_width) or (x_height < y_height):
            return False
    return True


def _is_crop(x_grids: list[Grid], y_grids: list[Grid])->bool:
    for x_grid, y_grid in zip(x_grids, y_grids):
        result = x_grid.find_subgrid(y_grid)
        if result is None:
            return False
    return True
