from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [
        [FilledRectangle(0, 0, 2, 2, 1), FilledRectangle(1, 1, 2, 2, 2)],
        [FilledRectangle(0, 0, 2, 2, 2), FilledRectangle(1, 1, 2, 2, 1)],
        [FilledRectangle(0, 0, 2, 2, 3), FilledRectangle(1, 1, 2, 2, 1)]]
    y_shapes = x_shapes
    action = DrawCanvas(
        width_model=ConstantModel(4),
        height_model=ConstantModel(4),
        layer_model=FunctionModel(
            lambda x: np.where(x['shape0.top_color'] > x['shape1.top_color'], 1, 0)))
    expect = [Grid([[1, 1, 0, 0], [1, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]]),
              Grid([[2, 2, 0, 0], [2, 2, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
              Grid([[3, 3, 0, 0], [3, 3, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])]

    state = create_test_state(x_shapes, y_shapes)
    state = state.update(has_layer=True)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    print(result.out)
    print(expect)
    assert result.out == expect
