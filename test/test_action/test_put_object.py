from .util import *


def test_basic():
    params = GlobalParams()
    all_x_shapes = [[Unknown(0, 0, x333_0.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, x333_1.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, x333_2.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, x333_3.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, x333_4.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, x333_5.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, x333_6.replace_color(0, NULL_COLOR))]]
    all_y_shapes = [[Unknown(0, 0, y333_0.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, y333_1.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, y333_2.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, y333_3.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, y333_4.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, y333_5.replace_color(0, NULL_COLOR))],
                    [Unknown(0, 0, y333_6.replace_color(0, NULL_COLOR))]]

    def selection(df):
        result = []
        for i, row in df.iterrows():
            if row['shape0.top_color'] == 1:
                result.append(1)
            elif row['shape0.top_color'] == 2:
                result.append(0)
            else:
                result.append(2)
        return result
    action = PutObject(FunctionModel(selection), ConstantModel(0),
                       ConstantModel(0), params)

    state = create_test_state(all_x_shapes, all_y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    for out_shapes, y_shapes in zip(result.out_shapes, all_y_shapes):
        assert out_shapes[1] == y_shapes[0]


y333_0 = Grid([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0]
])

y333_1 = Grid([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0]
])

y333_2 = Grid([
    [0, 0, 5],
    [0, 0, 5],
    [5, 5, 5]
])

y333_3 = Grid([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0]
])

y333_4 = Grid([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0]
])

y333_5 = Grid([
    [5, 5, 5],
    [0, 5, 0],
    [0, 5, 0]
])

y333_6 = Grid([
    [0, 0, 5],
    [0, 0, 5],
    [5, 5, 5]
])

x333_0 = Grid([
    [2, 0, 0, 0, 0],
    [0, 2, 0, 0, 2],
    [2, 0, 0, 2, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 2, 2, 0]
])

x333_1 = Grid([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 1, 0, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1]
])

x333_2 = Grid([
    [3, 0, 0, 0, 0],
    [0, 0, 0, 3, 3],
    [0, 3, 3, 0, 0],
    [0, 3, 0, 3, 0],
    [3, 0, 3, 3, 0]
])

x333_3 = Grid([
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])

x333_4 = Grid([
    [2, 0, 2, 0, 2],
    [2, 0, 0, 0, 2],
    [2, 2, 0, 0, 0],
    [2, 0, 0, 2, 2],
    [2, 2, 2, 0, 2]
])

x333_5 = Grid([
    [0, 2, 0, 2, 0],
    [0, 2, 2, 2, 0],
    [0, 2, 2, 0, 2],
    [2, 2, 2, 0, 0],
    [0, 0, 2, 0, 2]
])

x333_6 = Grid([
    [0, 3, 0, 3, 0],
    [3, 3, 0, 0, 0],
    [0, 3, 0, 0, 0],
    [0, 0, 3, 0, 0],
    [3, 3, 3, 0, 0]
])
