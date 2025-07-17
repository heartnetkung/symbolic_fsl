import pandas as pd
from ...base import *
from ...graphic import *
from ...manager.draw_line import *
from copy import deepcopy
from typing import Optional


def make_training_nav_df(
        state: ArcTrainingState, task: TrainingDrawLineTask)->Optional[pd.DataFrame]:
    assert state.out_shapes is not None

    updated_x_grids = [_draw_x_grid(grid, shapes).data
                       for grid, shapes in zip(state.x, state.out_shapes)]
    result = {'color': [], 'next_cell': [], 'next_2_cell': [],
              'left_cell': [], 'left_2_cell': [], 'right_cell': [], 'right_2_cell': []}

    y_lines = task.get_attention_aligned_lines()
    for id1, y_line in zip(task.atn.sample_index, y_lines):
        grid = Grid(deepcopy(updated_x_grids[id1]))
        blob = y_line.to_dir()
        if blob is None:
            return None

        _dir, navs = blob
        _handle_line(result, grid, y_line.color, _dir, y_line.coords[0], navs)
    return pd.DataFrame(result)


def _handle_line(result: dict[str, list], grid: Grid, color: int, init_dir: Direction,
                 init_coord: Coordinate, navigations: list[Navigation])->None:
    grid.safe_assign_c(init_coord, color)
    current_coord, current_dir = init_coord, init_dir

    for nav in navigations:
        append_step_df(result, grid, current_coord, current_dir, color)
        if nav == Navigation.turn_left:
            current_dir = current_dir.left()
        elif nav == Navigation.turn_right:
            current_dir = current_dir.right()

        current_coord = current_dir.proceed(current_coord)
        grid.safe_assign_c(current_coord, color)


def _draw_x_grid(grid: Grid, shapes: list[Shape])->Grid:
    canvas = make_grid(grid.width, grid.height)
    for shape in shapes:
        shape.draw(canvas)
    return canvas


def generate_step_df(
        grid: Grid, pos: Coordinate, dir_: Direction, color: int)->pd.DataFrame:
    result = {
        'color': [color],
        'next_cell': [grid.safe_access_c(dir_.proceed(pos, 1))],
        'next_2_cell': [grid.safe_access_c(dir_.proceed(pos, 2))],
        'left_cell': [grid.safe_access_c(dir_.left().proceed(pos, 1))],
        'left_2_cell': [grid.safe_access_c(dir_.left().proceed(pos, 2))],
        'right_cell': [grid.safe_access_c(dir_.right().proceed(pos, 1))],
        'right_2_cell': [grid.safe_access_c(dir_.right().proceed(pos, 2))]
    }
    return pd.DataFrame(result)


def append_step_df(result: dict[str, list], grid: Grid, pos: Coordinate,
                   dir_: Direction, color: int)->None:
    result['color'].append(color)
    result['next_cell'].append(grid.safe_access_c(dir_.proceed(pos, 1)))
    result['next_2_cell'].append(grid.safe_access_c(dir_.proceed(pos, 2)))
    result['left_cell'].append(grid.safe_access_c(dir_.left().proceed(pos, 1)))
    result['left_2_cell'].append(grid.safe_access_c(dir_.left().proceed(pos, 2)))
    result['right_cell'].append(grid.safe_access_c(dir_.right().proceed(pos, 1)))
    result['right_2_cell'].append(grid.safe_access_c(dir_.right().proceed(pos, 2)))
