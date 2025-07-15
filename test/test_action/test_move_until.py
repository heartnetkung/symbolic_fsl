from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(0, 0, 2, 2, 1), FilledRectangle(4, 1, 2, 2, 1)],
                [FilledRectangle(0, 0, 2, 2, 2), FilledRectangle(5, 0, 2, 2, 1)]]
    y_shapes = [[FilledRectangle(2, 0, 2, 2, 1), FilledRectangle(4, 1, 2, 2, 1)],
                [FilledRectangle(3, 0, 2, 2, 2), FilledRectangle(5, 0, 2, 2, 1)]]
    action = MoveUntil(MoveType.right, UntilType.touch, 0, 1)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_basic2():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(0, 0, 2, 2, 1), FilledRectangle(4, 1, 2, 2, 1)],
                [FilledRectangle(0, 0, 2, 2, 2), FilledRectangle(5, 0, 2, 2, 1)]]
    y_shapes = [[FilledRectangle(2, 0, 2, 2, 1), FilledRectangle(4, 1, 2, 2, 1)],
                [FilledRectangle(3, 0, 2, 2, 2), FilledRectangle(5, 0, 2, 2, 1)]]
    action = MoveUntil(MoveType.toward_1d, UntilType.touch, 0, 1)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_basic3():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(0, 0, 2, 2, 1), FilledRectangle(4, 1, 2, 2, 1)],
                [FilledRectangle(0, 0, 2, 2, 2), FilledRectangle(5, 0, 2, 2, 1)]]
    y_shapes = [[FilledRectangle(2, 0, 2, 2, 1), FilledRectangle(4, 1, 2, 2, 1)],
                [FilledRectangle(3, 0, 2, 2, 2), FilledRectangle(5, 0, 2, 2, 1)]]
    action = MoveUntil(MoveType.left, UntilType.touch, 0, 1)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result is None
