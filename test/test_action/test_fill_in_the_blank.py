from .util import *


def test_top_side():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [
        [Unknown(2, 2, Grid([
            [5,  -1, -1, -1, 5],
            [-1, 5, -1, 5, -1],
            [-1, 5, 5, 5, -1],
            [-1, 5, 5, 5, -1]]))],
        [Unknown(3, 3, Grid([
            [2,  -1, -1, -1, 2],
            [2, -1, -1, 2, 2],
            [2, 2, -1, 2, 2],
            [2, 2, 2, 2, -1]]))]]
    y_shapes = [
        [Unknown(2, 2, Grid([
            [5,  5, 5, 5, 5],
            [-1, 5, 5, 5, -1],
            [-1, 5, 5, 5, -1],
            [-1, 5, 5, 5, -1]]))],
        [Unknown(3, 3, Grid([
            [2,  2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, -1]]))]]
    action = FillInTheBlank(
        ExpansionMode.top_left, 0,
        ColumnModel('shape0.width'), ColumnModel('shape0.height'),
        FunctionModel(lambda x: np.where(
            x['is_topside(x,y)'], x['shape.top_color'], -1)),
        params
    )

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_inside():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [
        [Unknown(2, 2, Grid([
            [-1,  5, 5, -1, 5],
            [5, -1, -1, 5, -1],
            [-1, 5, 5, 5, -1],
            [-1, 5, 5, 5, -1]]))],
        [Unknown(3, 3, Grid([
            [2,  -1, -1, -1, 2],
            [2, 2, -1, 2, 2],
            [2, -1, 2, -1, 2],
            [2, 2, 2, 2, -1]]))]]
    y_shapes = [
        [Unknown(2, 2, Grid([
            [-1,  5, 5, -1, 5],
            [5, 4, 4, 5, -1],
            [-1, 5, 5, 5, -1],
            [-1, 5, 5, 5, -1]]))],
        [Unknown(3, 3, Grid([
            [2,  -1, -1, -1, 2],
            [2, 2, -1, 2, 2],
            [2, 4, 2, 4, 2],
            [2, 2, 2, 2, -1]]))]]
    action = FillInTheBlank(
        ExpansionMode.top_left, 0,
        ColumnModel('shape0.width'), ColumnModel('shape0.height'),
        FunctionModel(lambda x: np.where(
            x['is_outside(x,y)'], -1, 4)),
        params
    )

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_adjacent():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [
        [Unknown(2, 2, Grid([
            [-1,  5, -1],
            [5, 4, 5],
            [-1, 5, -1]]))],
        [Unknown(3, 3, Grid([
            [-1,  2, -1],
            [2, 3, 2],
            [-1, 2, -1]]))]]
    y_shapes = [
        [Unknown(1, 1, Grid([
            [-1,  -1, 5, -1, -1],
            [-1, 5,  5, 5, -1],
            [5, 5, 4, 5, 5],
            [-1, 5,  5, 5, -1],
            [-1,  -1, 5, -1, -1]]))],
        [Unknown(2, 2, Grid([
            [-1,  -1, 2, -1, -1],
            [-1, 2,  2, 2, -1],
            [2, 2, 3, 2, 2],
            [-1, 2,  2, 2, -1],
            [-1,  -1, 2, -1, -1]]))]]

    action = FillInTheBlank(
        ExpansionMode.center, 0,
        ConstantModel(5), ConstantModel(5),
        FunctionModel(lambda df: np.where(
            df['adjacent(x,y)'] >= 0, df['adjacent(x,y)'], -1)),
        params
    )

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_mirror():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [
        [Unknown(2, 2, Grid([
            [-1,  5, 5],
            [5, -1, -1],
            [-1, 5, 5],
            [-1, 5, 5]]))],
        [Unknown(3, 3, Grid([
            [2,  -1, -1],
            [2, 2, -1],
            [2, -1, 2],
            [2, 2, 2]]))]]
    y_shapes = [
        [Unknown(2, 2, Grid([
            [-1,  5, 5, 5, -1],
            [5, -1, -1, -1, 5],
            [-1, 5, 5, 5, -1],
            [-1, 5, 5, 5, -1]]))],
        [Unknown(3, 3, Grid([
            [2,  -1, -1, -1, 2],
            [2, 2, -1, 2, 2],
            [2, -1, 2, -1, 2],
            [2, 2, 2, 2, 2]]))]]
    action = FillInTheBlank(
        ExpansionMode.top_left, 0,
        ConstantModel(5), ConstantModel(4),
        ColumnModel('cell(-x,y)'),
        params
    )

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_mirror2():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [
        [Unknown(2, 2, Grid([
            [3, -1, 8, -1, -1],
            [-1, 2, -1, 2, -1],
            [8, -1, 3, -1, 8],
            [-1, 2, -1, 2, -1],
            [-1, -1, 8, -1, -1]]))],
        [Unknown(3, 3, Grid([
            [8, -1, 8, -1, 8],
            [-1, 4, -1, -1, -1],
            [8, -1, 1, -1, 8],
            [-1, -1, -1, -1, -1],
            [8, -1, 8, -1, 8]]))]]
    y_shapes = [
        [Unknown(2, 2, Grid([
            [3, -1, 8, -1, 3],
            [-1, 2, -1, 2, -1],
            [8, -1, 3, -1, 8],
            [-1, 2, -1, 2, -1],
            [3, -1, 8, -1, 3]]))],
        [Unknown(3, 3, Grid([
            [8, -1, 8, -1, 8],
            [-1, 4, -1, 4, -1],
            [8, -1, 1, -1, 8],
            [-1, 4, -1, 4, -1],
            [8, -1, 8, -1, 8]]))]]
    action = FillInTheBlank(
        ExpansionMode.top_left, 0,
        ConstantModel(5), ConstantModel(5),
        ColumnModel('mirror(x,y)'),
        params
    )

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_star():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [
        [Unknown(2, 2, Grid([
            [-1,  5, -1],
            [5, 4, 5],
            [-1, 5, -1]]))],
        [Unknown(3, 3, Grid([
            [-1,  2, -1],
            [2, 3, 2],
            [-1, 2, -1]]))]]
    y_shapes = [
        [Unknown(1, 1, Grid([
            [4,  -1, 5, -1, 4],
            [-1, 4,  5, 4, -1],
            [5, 5, 4, 5, 5],
            [-1, 4,  5, 4, -1],
            [4,  -1, 5, -1, 4]]))],
        [Unknown(2, 2, Grid([
            [3,  -1, 2, -1, 3],
            [-1, 3,  2, 3, -1],
            [2, 2, 3, 2, 2],
            [-1, 3,  2, 3, -1],
            [3,  -1, 2, -1, 3]]))]]

    def redraw(df: pd.DataFrame):
        result = np.where(df['is_plus_path(x,y)'] == 1, df['shape.top_color'], -1)
        result2 = np.where(df['is_cross_path(x,y)'] == 1,
                           df['shape.least_color'], result)
        return result2

    action = FillInTheBlank(
        ExpansionMode.center, 0,
        ConstantModel(5), ConstantModel(5),
        FunctionModel(redraw),
        params
    )

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_tile():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [
        [Unknown(0, 0, Grid([
            [0,  1, 0],
            [1,  1, 0],
            [0,  1, 0],
            [0,  1, 1],
            [0,  1, 0],
            [1,  1, 0]]))],
        [Unknown(0, 0, Grid([
            [0,  1, 0],
            [1,  0, 1],
            [0,  1, 0],
            [1,  0, 1],
            [0,  1, 0],
            [1,  0, 1]]))],
        [Unknown(0, 0, Grid([
            [0,  1, 0],
            [1,  1, 0],
            [0,  1, 0],
            [0,  1, 0],
            [1,  1, 0],
            [0,  1, 0]]))]]
    y_shapes = [
        [Unknown(0, 0, Grid([
            [0,  1, 0],
            [1,  1, 0],
            [0,  1, 0],
            [0,  1, 1],
            [0,  1, 0],
            [1,  1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 0]]))],
        [Unknown(0, 0, Grid([
            [0,  1, 0],
            [1,  0, 1],
            [0,  1, 0],
            [1,  0, 1],
            [0,  1, 0],
            [1,  0, 1],
            [0,  1, 0],
            [1,  0, 1],
            [0,  1, 0]]))],
        [Unknown(0, 0, Grid([
            [0,  1, 0],
            [1,  1, 0],
            [0,  1, 0],
            [0,  1, 0],
            [1,  1, 0],
            [0,  1, 0],
            [0,  1, 0],
            [1,  1, 0],
            [0,  1, 0]]))]]

    def redraw(df: pd.DataFrame):
        return np.where(df['tile(x,y)'] == 1, 1, 0)
    action = FillInTheBlank(
        ExpansionMode.top_left, 0,
        ConstantModel(3), ConstantModel(9),
        FunctionModel(redraw),
        params
    )

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes


def test_global_features():
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_shapes = [
        [Unknown(0, 0, Grid([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [8, -1, -1, -1, 2, 2, -1, 2, 2, 2, 2, 2],
            [8, -1, -1, -1, -1, 2, -1, -1, 2, 2, -1, 2],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, 8, -1, -1, -1, -1, 8, 8, -1, -1, -1, 8],
            [8, 8, 8, -1, -1, 8, 8, 8, -1, -1, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]
        ))],
        [Unknown(0, 0, Grid([
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [2, -1, -1, -1, -1, -1, 8, 8, 8, 8, 8, 8],
            [2, 2, -1, -1, -1, -1, -1, 8, 8, -1, -1, 8],
            [2, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, 8],
            [2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2],
            [2, 2, -1, 2, -1, -1, 2, -1, -1, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ]))],
        [Unknown(0, 0, Grid([
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 2],
            [8, 8, 8, -1, 8, 8, -1, 8, -1, 2],
            [8, 8, -1, -1, 8, -1, -1, -1, -1, 2],
            [8, 8, -1, -1, -1, -1, -1, -1, 2, 2],
            [8, -1, -1, -1, -1, -1, -1, -1, 2, 2],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [8, -1, -1, -1, -1, -1, 2, 2, -1, 2],
            [8, 2, -1, -1, -1, 2, 2, 2, 2, 2],
            [8, 2, 2, 2, 2, 2, 2, 2, 2, 2]]))]]
    y_shapes = [
        [Unknown(0, 0, Grid([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [8, -1, -1, 3, 2, 2, -1, 2, 2, 2, 2, 2],
            [8, -1, -1, 3, -1, 2, -1, -1, 2, 2, -1, 2],
            [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
            [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
            [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
            [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
            [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8],
            [8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8],
            [8, 8, -1, 3, -1, -1, 8, 8, -1, -1, -1, 8],
            [8, 8, 8, 3, -1, 8, 8, 8, -1, -1, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]))],
        [Unknown(0, 0, Grid([
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [2, -1, -1, -1, 3, 3, 8, 8, 8, 8, 8, 8],
            [2, 2, -1, -1, 3, 3, -1, 8, 8, -1, -1, 8],
            [2, -1, -1, -1, 3, 3, -1, 8, -1, -1, -1, 8],
            [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8],
            [2, 2, 2, -1, 3, 3, -1, -1, -1, -1, -1, 8],
            [2, 2, -1, -1, 3, 3, -1, -1, -1, -1, -1, 8],
            [2, 2, -1, -1, 3, 3, -1, -1, -1, -1, -1, 8],
            [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8],
            [2, -1, -1, -1, 3, 3, -1, -1, -1, -1, 2, 2],
            [2, 2, -1, 2, 3, 3, 2, -1, -1, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ]))],
        [Unknown(0, 0, Grid([
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 2],
            [8, 8, 8, 3, 8, 8, -1, 8, -1, 2],
            [8, 8, -1, 3, 8, -1, -1, -1, -1, 2],
            [8, 8, -1, 3, -1, -1, -1, -1, 2, 2],
            [8, -1, -1, 3, -1, -1, -1, -1, 2, 2],
            [8, 3, 3, 3, 3, 3, 3, 3, 3, 2],
            [8, 3, 3, 3, 3, 3, 3, 3, 3, 2],
            [8, -1, -1, 3, -1, -1, 2, 2, -1, 2],
            [8, 2, -1, 3, -1, 2, 2, 2, 2, 2],
            [8, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ]))]]

    def redraw(df: pd.DataFrame):
        cond1 = df['row_blank_count_rank(x,y)'] == 0
        cond2 = df['col_blank_count_rank(x,y)'] == 0
        return np.where(np.logical_or(cond1, cond2), 3, -1)
    action = FillInTheBlank(
        ExpansionMode.top_left, 0,
        ColumnModel('shape0.width'), ColumnModel('shape0.height'),
        FunctionModel(redraw),
        params
    )

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes
