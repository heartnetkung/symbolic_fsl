from .util import *


def test_include_corder():
    params = GlobalParams()

    def func(df):
        result = np.where(df['cell(x,y)'] == 8, 1, 0)
        second_cond = np.logical_or(df['cell(x,y-1)'] == 5, df['cell(x,y+1)'] == 5)
        return np.where(second_cond, result, 0)

    action = Crop(BoundScanModel(FunctionModel(func), BoundScan(True)), params)
    state = ArcTrainingState(
        [x1], [y1], None,
        [0], [0], False, 0,
        [[Unknown(0, 0, x1)]], [[Unknown(0, 0, y1)]], [[Unknown(0, 0, x1)]]
    ).check_all()
    program = AttentionExpertProgram(action, params, 1)
    result = program.run(state)
    assert result.out_shapes == [[Unknown(0, 0, y1)]]


def test_exclude_corder():
    params = GlobalParams()

    def func(df):
        return np.where(df['cell(x,y)'] == 4, 1, 0)
    action = Crop(BoundScanModel(FunctionModel(func), BoundScan(False)), params)

    state = ArcTrainingState(
        [x2], [y2], None,
        [0], [0], False, 0,
        [[Unknown(0, 0, x2)]], [[Unknown(0, 0, y2)]], [[Unknown(0, 0, x2)]]
    ).check_all()
    program = AttentionExpertProgram(action, params, 1)
    result = program.run(state)
    assert result.out_shapes == [[Unknown(0, 0, y2)]]


x1 = Grid([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 8, 0, 0, 0, 8, 0, 0, 8],
           [0, 5, 0, 0, 0, 5, 0, 0, 0],
           [0, 5, 0, 8, 0, 5, 0, 8, 0],
           [0, 5, 0, 0, 0, 5, 0, 0, 0],
           [0, 8, 0, 0, 0, 8, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 8, 0, 0, 0, 8, 0],
           [0, 8, 0, 0, 0, 0, 0, 0, 0]])
y1 = Grid([[8, 0, 0, 0, 8],
           [5, 0, 0, 0, 5],
           [5, 0, 8, 0, 5],
           [5, 0, 0, 0, 5],
           [8, 0, 0, 0, 8]])
x2 = Grid([[0, 0, 0, 0, 0, 0, 0],
           [0, 4, 0, 0, 0, 4, 0],
           [0, 0, 0, 2, 0, 0, 0],
           [0, 0, 2, 2, 2, 0, 0],
           [0, 0, 0, 2, 2, 0, 0],
           [0, 4, 0, 0, 0, 4, 0],
           [0, 0, 0, 0, 0, 0, 0]])
y2 = Grid([[0, 2, 0],
           [2, 2, 2],
           [0, 2, 2]])
