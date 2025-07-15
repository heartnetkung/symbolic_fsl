from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(0, 2, 1, 1, 1), FilledRectangle(0, 0, 1, 1, 2)],
                [FilledRectangle(0, 2, 1, 1, 2), FilledRectangle(0, 0, 1, 1, 3)],
                [FilledRectangle(0, 2, 1, 1, 3), FilledRectangle(0, 0, 1, 1, 4)]]
    y_shapes = [[FilledRectangle(0, 0, 1, 1, 2)],
                [FilledRectangle(0, 0, 1, 1, 3)],
                [FilledRectangle(0, 0, 1, 1, 4)]]
    action = CleanUp(FunctionModel(
        lambda x: np.where(x['shape0.y'] == 0, 1, 0)), params)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes
