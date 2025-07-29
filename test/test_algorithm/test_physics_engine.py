from ...arc.base import *
from ...arc.graphic import *
from ...arc.algorithm.physics_engine import *


def test_basic():
    width, height, dir_ = 10, 10, Direction.north
    still_shapes = [Unknown(1, 0, Grid([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, 1, 1, 1, 1, -1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, -1, 1, -1, -1, -1, 1, -1, 1],
        [-1, -1, 1, -1, -1, -1, -1, -1, 1]
    ]))]
    moving_shapes = [
        FilledRectangle(2, 7, 1, 3, 2),
        FilledRectangle(4, 8, 1, 1, 2),
        FilledRectangle(4, 9, 1, 1, 2),
        FilledRectangle(5, 6, 1, 4, 2),
        FilledRectangle(6, 9, 1, 1, 2),
        FilledRectangle(8, 4, 1, 6, 2),
    ]
    simulator = SolidSimulation(width, height, moving_shapes, still_shapes, dir_)
    output_shapes = simulator.simulate()
    assert output_shapes is not None
    canvas = draw_canvas(width, height, output_shapes+still_shapes, 0)
    canvas.print_grid2()
    assert canvas == expect1


expect1 = Grid([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 2, 1, 1, 1, 1, 1, 2, 1],
    [0, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 2, 1, 2, 2, 0, 1, 2, 1],
    [0, 0, 0, 1, 0, 2, 0, 0, 2, 1],
    [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
