from .util import *


def test_basic():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [[Unknown(2, 2, grid.flip_h())],
                [Unknown(3, 3, grid.flip_v())],
                [Unknown(4, 4, grid.flip_both())],
                [Unknown(5, 5, grid.transpose())],
                [Unknown(6, 6, grid.transpose().flip_h())],
                [Unknown(7, 7, grid.transpose().flip_v())],
                [Unknown(8, 8, grid.transpose().flip_both())]]
    y_shapes = [[Unknown(2, 2, grid)],
                [Unknown(3, 3, grid)],
                [Unknown(4, 4, grid)],
                [Unknown(5, 5, grid)],
                [Unknown(6, 6, grid)],
                [Unknown(7, 7, grid)],
                [Unknown(8, 8, grid)]]
    action = GeomTransform(FunctionModel(lambda x: x['shape0.x']-1), params)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes
