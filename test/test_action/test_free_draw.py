from .util import *


def _run(x_shape: Shape, y_shape: Shape, action: FreeDraw, params: GlobalParams)->None:
    x_shapes = [[x_shape]]
    y_shapes = [[y_shape]]
    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    print_pair(result)
    assert result.out_shapes == y_shapes


def test_0():
    def func(df):
        return np.where(df['cell(+x/w,y/h)'] != 0, df[f'cell(+x%w,y%h)'], 0)
    params, param = GlobalParams(), FreeDrawParam.normal
    action = FreeDraw(param, ConstantModel(9), ConstantModel(9),
                      FunctionModel(func), params)
    x = Unknown(0, 0, x0)
    y = Unknown(0, 0, y0)
    _run(x, y, action, params)


def test_105():
    def func(df):
        return df['quadrant_cw_rotate(+x,y)']
    params, param = GlobalParams(), FreeDrawParam.normal
    action = FreeDraw(param, ConstantModel(6), ConstantModel(6),
                      FunctionModel(func), params)
    x = Unknown(0, 0, x105)
    y = Unknown(0, 0, y105)
    _run(x, y, action, params)


def test_265():
    def func(df):
        result = []
        for i, row in df.iterrows():
            if row['cell(x-1,y+1)'] == 2:
                result.append(6)
            elif row['cell(x-1,y-1)'] == 2:
                result.append(7)
            elif row['cell(x+1,y+1)'] == 2:
                result.append(3)
            elif row['cell(x+1,y-1)'] == 2:
                result.append(8)
            else:
                result.append(0)
        return result
    params, param = GlobalParams(), FreeDrawParam.normal
    action = FreeDraw(param, ConstantModel(5), ConstantModel(3),
                      FunctionModel(func), params)
    x = Unknown(0, 0, x265)
    y = Unknown(0, 0, y265)
    _run(x, y, action, params)


def test_303():
    def func(df):
        result = []
        for i, row in df.iterrows():
            if row['cell(+x/w,y/h)'] == 1:
                result.append(row[f'cell(+x%w,y%h)'])
            else:
                result.append(0)
        return result
    params, param = GlobalParams(), FreeDrawParam.normal
    action = FreeDraw(param, ConstantModel(9), ConstantModel(9),
                      FunctionModel(func), params)
    x = Unknown(0, 0, x303)
    y = Unknown(0, 0, y303)
    _run(x, y, action, params)


def test_342():
    def func(df):
        return df['tile(x,y)']
    params, param = GlobalParams(), FreeDrawParam.normal
    action = FreeDraw(param, ConstantModel(15), ConstantModel(5),
                      FunctionModel(func), params)
    x = Unknown(0, 0, x342)
    y = Unknown(0, 0, y342)
    _run(x, y, action, params)


def test_194():
    def func(df):
        result = []
        for i, row in df.iterrows():
            if row['cell(x,y)'] == 5:
                result.append(row['unscaled_cell(+x,y)'])
            else:
                result.append(0)
        return result
    params, param = GlobalParams(), FreeDrawParam.normal
    action = FreeDraw(param, ConstantModel(9), ConstantModel(9),
                      FunctionModel(func), params)
    x = Unknown(0, 0, x194)
    y = Unknown(0, 0, y194)
    _run(x, y, action, params)


x0 = Grid([
    [0, 7, 7],
    [7, 7, 7],
    [0, 7, 7]
])
y0 = Grid([
    [0, 0, 0, 0, 7, 7, 0, 7, 7],
    [0, 0, 0, 7, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 7, 7, 0, 7, 7],
    [0, 7, 7, 0, 7, 7, 0, 7, 7],
    [7, 7, 7, 7, 7, 7, 7, 7, 7],
    [0, 7, 7, 0, 7, 7, 0, 7, 7],
    [0, 0, 0, 0, 7, 7, 0, 7, 7],
    [0, 0, 0, 7, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 7, 7, 0, 7, 7]
])

x105 = Grid([
    [6, 9, 9],
    [6, 4, 4],
    [6, 4, 4]
])
y105 = Grid([
    [6, 9, 9, 6, 6, 6],
    [6, 4, 4, 4, 4, 9],
    [6, 4, 4, 4, 4, 9],
    [9, 4, 4, 4, 4, 6],
    [9, 4, 4, 4, 4, 6],
    [6, 6, 6, 9, 9, 6]
])

x265 = Grid([
    [0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
y265 = Grid([
    [3, 0, 6, 0, 0],
    [0, 0, 0, 0, 0],
    [8, 0, 7, 0, 0]
])

x342 = Grid([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 2, 2, 0, 6, 2, 2, 0, 6, 2, 0, 0, 0, 0, 0],
    [6, 6, 2, 3, 6, 6, 2, 3, 6, 6, 0, 0, 0, 0, 0]
])
y342 = Grid([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2],
    [6, 2, 2, 0, 6, 2, 2, 0, 6, 2, 2, 0, 6, 2, 2],
    [6, 6, 2, 3, 6, 6, 2, 3, 6, 6, 2, 3, 6, 6, 2]
])

x303 = Grid([
    [1, 1, 7],
    [7, 4, 1],
    [5, 1, 7]
])
y303 = Grid([
    [1, 1, 7, 1, 1, 7, 0, 0, 0],
    [7, 4, 1, 7, 4, 1, 0, 0, 0],
    [5, 1, 7, 5, 1, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 7],
    [0, 0, 0, 0, 0, 0, 7, 4, 1],
    [0, 0, 0, 0, 0, 0, 5, 1, 7],
    [0, 0, 0, 1, 1, 7, 0, 0, 0],
    [0, 0, 0, 7, 4, 1, 0, 0, 0],
    [0, 0, 0, 5, 1, 7, 0, 0, 0]
])

x194 = Grid([
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 5, 5, 5, 0, 0, 0],
    [5, 5, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 0, 0, 0, 5, 5, 5]
])
y194 = Grid([
    [5, 5, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 0, 0, 5, 0, 0, 5, 0],
    [5, 0, 5, 5, 0, 5, 5, 0, 5],
    [0, 0, 0, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 5, 0, 0, 0],
    [5, 5, 5, 0, 0, 0, 5, 5, 5],
    [0, 5, 0, 0, 0, 0, 0, 5, 0],
    [5, 0, 5, 0, 0, 0, 5, 0, 5]
])
