from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[Diagonal(0, 0, 2, 1, True)],
                [Diagonal(0, 0, 2, 2, True)],
                [Diagonal(0, 0, 2, 3, False)]]
    y_shapes = [[Diagonal(0, 0, 2, 1, True), Diagonal(2, 2, 2, 1, False)],
                [Diagonal(0, 0, 2, 2, True), Diagonal(2, 2, 2, 2, False)],
                [Diagonal(0, 0, 2, 3, False), Diagonal(2, 2, 2, 3, True)]]
    action = CreateDiagonal(
        x_model=ConstantModel(2),
        y_model=ConstantModel(2),
        width_model=ConstantModel(2),
        color_model=ColumnModel('shape0.top_color'),
        orientation_model=FunctionModel(
            lambda x: np.where(x['shape0.north_west'], 0, 1)),
        params=params)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes
