from ...arc.base import *
from ...arc.graphic import *
from ...arc.algorithm.solve_stack import *
from ...arc.algorithm.find_background import *


def do_test(grid: Grid, bg: int, output: list[Shape]):
    grid2 = grid.replace_color(bg, NULL_COLOR)
    shapes = list_objects(grid2, True)
    result = solve_stack(shapes, grid2)
    assert result == output


def test_tile_stack():
    ds = read_datasets(DatasetChoice.train_v2)

    problem = 16
    y = ds[problem].y_train
    state = ds[problem].to_training_state()
    _, y_model = find_backgrounds(state)[0]
    y_backgrounds = y_model.predict_int(make_background_df(state))
    outputs = [
        [FilledRectangle(1, 14, 6, 1, 8),
         FilledRectangle(2, 4, 6, 1, 3),
         FilledRectangle(2, 8, 4, 1, 7),
         FilledRectangle(3, 2, 1, 9, 4),
         FilledRectangle(5, 12, 1, 7, 9)],
        [FilledRectangle(2, 20, 6, 1, 5),
         FilledRectangle(3, 6, 9, 1, 3),
         FilledRectangle(4, 18, 1, 10, 6),
         FilledRectangle(6, 2, 1, 12, 2),
         FilledRectangle(14, 12, 1, 6, 8)]
    ]
    for grid, bg, output in zip(y, y_backgrounds, outputs):
        do_test(grid, bg, output)

    problem = 46
    y = ds[problem].y_train
    state = ds[problem].to_training_state()
    _, y_model = find_backgrounds(state)[0]
    y_backgrounds = y_model.predict_int(make_background_df(state))
    outputs = [
        [FilledRectangle(2, 7, 14, 4, 7),
         FilledRectangle(6, 4, 10, 4, 3),
         FilledRectangle(11, 1, 5, 2, 2),
         FilledRectangle(16, 0, 1, 16, 5)],
        [FilledRectangle(0, 6, 4, 9, 8),
         FilledRectangle(0, 15, 15, 1, 5),
         FilledRectangle(6, 3, 5, 12, 4),
         FilledRectangle(7, 11, 3, 4, 3)],
        [FilledRectangle(0, 0, 1, 17, 5),
         FilledRectangle(1, 1, 9, 6, 3),
         FilledRectangle(1, 5, 13, 8, 6),
         FilledRectangle(1, 10, 6, 2, 2)]
    ]
    for grid, bg, output in zip(y, y_backgrounds, outputs):
        do_test(grid, bg, output)

    problem = 40
    X = ds[problem].X_train
    state = ds[problem].to_training_state()
    x_model, _ = find_backgrounds(state)[0]
    x_backgrounds = x_model.predict_int(make_background_df(state))
    outputs = [
        [FilledRectangle(4, 0, 1, 11, 8),
         FilledRectangle(0, 3, 11, 1, 8),
         FilledRectangle(0, 9, 11, 1, 8)],
        [FilledRectangle(1, 0, 1, 15, 7),
         FilledRectangle(10, 0, 1, 15, 7),
         FilledRectangle(13, 0, 1, 15, 7),
         FilledRectangle(0, 2, 15, 1, 7)],
        [FilledRectangle(6, 0, 1, 27, 1),
         FilledRectangle(21, 0, 1, 27, 1),
         FilledRectangle(23, 0, 1, 27, 1),
         FilledRectangle(25, 0, 1, 27, 1),
         FilledRectangle(0, 2, 27, 1, 1),
         FilledRectangle(0, 7, 27, 1, 1),
         FilledRectangle(0, 16, 27, 1, 1),
         FilledRectangle(0, 21, 27, 1, 1),
         FilledRectangle(0, 23, 27, 1, 1)]]
    for grid, bg, output in zip(X, x_backgrounds, outputs):
        do_test(grid, bg, output)

    problem = 50
    y = ds[problem].y_train
    state = ds[problem].to_training_state()
    _, y_model = find_backgrounds(state)[0]
    y_backgrounds = y_model.predict_int(make_background_df(state))
    outputs = [
        [FilledRectangle(3, 0, 1, 9, 1),
         FilledRectangle(6, 0, 1, 9, 1),
         FilledRectangle(0, 3, 9, 1, 1),
         FilledRectangle(0, 6, 9, 1, 1),
         FilledRectangle(2, 2, 3, 3, 3),
         FilledRectangle(5, 5, 3, 3, 3),
         FilledRectangle(3, 3, 1, 1, 2),
         FilledRectangle(6, 6, 1, 1, 2)],
        [FilledRectangle(0, 1, 13, 1, 1),
         FilledRectangle(0, 3, 13, 1, 1),
         FilledRectangle(0, 6, 13, 1, 1),
         FilledRectangle(0, 9, 13, 1, 1),
         FilledRectangle(2, 0, 1, 13, 1),
         FilledRectangle(5, 0, 1, 13, 1),
         FilledRectangle(9, 0, 1, 13, 1),
         FilledRectangle(11, 0, 1, 13, 1),
         FilledRectangle(10, 0, 3, 3, 3),
         FilledRectangle(1, 2, 3, 3, 3),
         FilledRectangle(8, 5, 3, 3, 3),
         FilledRectangle(4, 8, 3, 3, 3),
         FilledRectangle(11, 1, 1, 1, 2),
         FilledRectangle(2, 3, 1, 1, 2),
         FilledRectangle(9, 6, 1, 1, 2),
         FilledRectangle(5, 9, 1, 1, 2)],
        [FilledRectangle(2, 0, 1, 11, 1),
         FilledRectangle(6, 0, 1, 11, 1),
         FilledRectangle(8, 0, 1, 11, 1),
         FilledRectangle(0, 1, 11, 1, 1),
         FilledRectangle(0, 4, 11, 1, 1),
         FilledRectangle(0, 9, 11, 1, 1),
         FilledRectangle(7, 0, 3, 3, 3),
         FilledRectangle(1, 3, 3, 3, 3),
         FilledRectangle(5, 8, 3, 3, 3),
         FilledRectangle(8, 1, 1, 1, 2),
         FilledRectangle(2, 4, 1, 1, 2),
         FilledRectangle(6, 9, 1, 1, 2)]
    ]
    for grid, bg, output in zip(y, y_backgrounds, outputs):
        do_test(grid, bg, output)


def test_361():
    raw_y_grids = [y361_0, y361_1, y361_2]
    y_grids = []
    for grid in raw_y_grids:
        y_grids.extend(geom_transform_all(grid.remove_bg()))
    all_y_shapes = [list_objects(grid) for grid in y_grids]
    for grid, shapes in zip(y_grids, all_y_shapes):
        result_shapes = solve_stack(shapes, grid, True)
        for result_shape in result_shapes:
            assert (result_shape.width == 10) or (result_shape.height == 10)


y361_0 = Grid([
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0]
])

y361_1 = Grid([
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

y361_2 = Grid([
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0]
])
