from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(0, 0, 1, 1, 1)],
                [FilledRectangle(0, 0, 1, 1, 2)],
                [FilledRectangle(0, 0, 1, 1, 3)]]
    y_shapes = [[FilledRectangle(0, 0, 1, 1, 2)],
                [FilledRectangle(0, 0, 1, 1, 3)],
                [FilledRectangle(0, 0, 1, 1, 4)]]
    action = Colorize(FunctionModel(lambda x: x['shape0.top_color']+1), params)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_unknown():
    params = GlobalParams()
    x_shapes = [[Unknown(0, 0, Grid([[1, -1, 1], [-1, 1, -1], [1, -1, 1]]))],
                [FilledRectangle(0, 0, 1, 1, 2)],
                [FilledRectangle(0, 0, 1, 1, 3)]]
    y_shapes = [[Unknown(0, 0, Grid([[2, -1, 2], [-1, 2, -1], [2, -1, 2]]))],
                [FilledRectangle(0, 0, 1, 1, 3)],
                [FilledRectangle(0, 0, 1, 1, 4)]]
    action = Colorize(FunctionModel(lambda x: x['shape0.top_color']+1), params)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes
