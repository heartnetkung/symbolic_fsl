from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(1, 1, 2, 2, 1)],
                [FilledRectangle(2, 2, 2, 2, 2)]]
    y_shapes = [[FilledRectangle(2, 0, 2, 2, 1)],
                [FilledRectangle(3, 0, 2, 2, 2)]]
    action = Move(FunctionModel(lambda x:x['shape0.x']+1), ConstantModel(0), 0)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes
