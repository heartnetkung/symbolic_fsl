from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(0, 0, 1, 1, 1)],
                [FilledRectangle(0, 0, 1, 1, 2)],
                [FilledRectangle(0, 0, 1, 1, 3)]]
    y_shapes = [[FilledRectangle(0, 0, 1, 1, 1), FilledRectangle(0, 0, 1, 1, 2)],
                [FilledRectangle(0, 0, 1, 1, 2), FilledRectangle(0, 0, 1, 1, 3)],
                [FilledRectangle(0, 0, 1, 1, 3), FilledRectangle(0, 0, 1, 1, 4)]]
    action = CreateRectangle(
        x_model=ConstantModel(0),
        y_model=ConstantModel(0),
        width_model=ConstantModel(1),
        height_model=ConstantModel(1),
        color_model=FunctionModel(lambda x: x['shape0.top_color']+1),
        params=params)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes
