from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(0, 0, 1, 1, 1)],
                [FilledRectangle(0, 0, 1, 1, 2)],
                [FilledRectangle(0, 0, 1, 1, 3)],
                [FilledRectangle(0, 0, 1, 1, 4)]]
    y_shapes = [[FilledRectangle(0, 0, 1, 1, 1), Diagonal(5, 0, 30, 1, True)],
                [FilledRectangle(0, 0, 1, 1, 2), Diagonal(0, 5, 30, 2, True)],
                [FilledRectangle(0, 0, 1, 1, 3), Diagonal(0, 0, 5, 3, False)],
                [FilledRectangle(0, 0, 1, 1, 4), Diagonal(0, 0, 15, 4, False)]]
    action = CreateBoundlessDiagonal(
        y_intercept_model=FunctionModel(mapping_func),
        color_model=ColumnModel('shape0.top_color'),
        orientation_model=FunctionModel(
            lambda x: np.where(x['shape0.top_color'].to_numpy() < 3, 1, 0)),
        params=params)

    state = create_test_state(x_shapes, y_shapes, 10, 10)
    assert state.y[0] == Grid([
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert state.y[1] == Grid([
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0], ])
    assert state.y[2] == Grid([
        [3, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert state.y[3] == Grid([
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0]])

    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def mapping_func(df):
    mapping = {1: -5, 2: 5, 3: 4, 4: 14}
    result = df['shape0.top_color'].to_numpy().copy()
    result[result == list(mapping.keys())] = list(mapping.values())
    return result
