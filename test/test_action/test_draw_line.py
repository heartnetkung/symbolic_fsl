from .util import *
from ...arc.manager.draw_line import *
from ...arc.expert.custom import *


def test_basic():
    params = GlobalParams()
    x = [make_grid(shapes[0].width, shapes[0].height, 0)
         for shapes in all_x_shapes]
    state = create_test_state(all_x_shapes, all_y_shapes)
    state = state.update(x=x, y=x)

    def nav(df):
        nc, n2c = df['next_cell'][0], df['next_2_cell'][0]
        lc, l2c = df['left_cell'][0], df['left_2_cell'][0]
        rc, r2c = df['right_cell'][0], df['right_2_cell'][0]

        turn = (nc in (MISSING_VALUE, 3)) or (n2c == 3)
        not_left = (lc in (MISSING_VALUE, 3)) or (l2c == 3)
        not_right = (rc in (MISSING_VALUE, 3)) or (r2c == 3)

        if not turn:
            return [Navigation.proceed.value]
        if not_left and not_right:
            return [Navigation.stop.value]
        if not_left:
            return [Navigation.turn_right.value]
        return [Navigation.turn_left.value]

    action = DrawLine(init_x_model=ConstantModel(0),
                      init_y_model=ConstantModel(0),
                      dir_model=ConstantModel(2),  # east
                      nav_model=FunctionModel(nav),
                      color_model=ConstantModel(3),
                      params=params)
    program = AttentionExpertProgram(action, params, 1)
    result = program.run(state)
    for out_shapes, y_shapes in zip(result.out_shapes, all_y_shapes):
        assert out_shapes[1] == y_shapes[0]


def test_expert_action():
    params = GlobalParams()
    x = [make_grid(shapes[0].width, shapes[0].height, 0)
         for shapes in all_x_shapes]
    state = create_test_state(all_x_shapes, all_y_shapes)
    state = state.update(x=x, y=x)

    tasks = make_line_tasks(state, params)
    expert = DrawLineExpert(params)
    actions = expert.solve_problem(state, tasks[0])
    result = actions[0].perform(state, tasks[0])
    for out_shapes, y_shapes in zip(result.out_shapes, all_y_shapes):
        assert out_shapes[1] == y_shapes[0]

    infer_actions = actions[0].train_models(state, tasks[0])
    result2 = infer_actions[0].perform(state, tasks[0])
    for out_shapes, y_shapes in zip(result2.out_shapes, all_y_shapes):
        assert out_shapes[1] == y_shapes[0]


# 57
all_y_shapes = [
    [Unknown(0, 0, Grid([
        [3, 3, 3, 3, 3, 3],
        [-1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, -1, 3],
        [3, -1, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3]
    ]))],
    [Unknown(0, 0, Grid([
        [3, 3, 3, 3, 3, 3, 3, 3],
        [-1, -1, -1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, 3, -1, 3],
        [3, -1, 3, 3, -1, 3, -1, 3],
        [3, -1, 3, 3, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3, 3, 3]
    ]))],
    [Unknown(0, 0, Grid([
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, 3],
        [3, -1, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, -1, -1, -1, -1, -1, -1, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, 3, 3, 3, 3, -1, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, -1, -1, -1, 3, -1, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, -1, 3, 3, 3, -1, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, -1, -1, -1, -1, -1, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, 3, 3, 3, 3, 3, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, 3],
        [3, -1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    ]))],
    [Unknown(0, 0, Grid([
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, 3],
        [3, -1, 3, 3, 3, 3, 3, 3, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, -1, -1, -1, -1, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, 3, 3, -1, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, -1, -1, -1, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, 3, 3, 3, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, -1, -1, -1, -1, -1, -1, 3, -1, 3],
        [3, -1, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    ]))],
    [Unknown(0, 0, Grid([
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, -1, -1, 3, -1, 3],
        [3, -1, 3, 3, 3, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, 3, 3, -1, 3, -1, 3],
        [3, -1, 3, -1, -1, -1, -1, 3, -1, 3],
        [3, -1, 3, 3, 3, 3, 3, 3, -1, 3],
        [3, -1, -1, -1, -1, -1, -1, -1, -1, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    ]))]
]
all_x_shapes = [
    [FilledRectangle(0, 0, 6, 6, 0)], [FilledRectangle(0, 0, 8, 8, 0)],
    [FilledRectangle(0, 0, 15, 15, 0)], [FilledRectangle(0, 0, 13, 13, 0)],
    [FilledRectangle(0, 0, 10, 10, 0)]]
