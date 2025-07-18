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
            lambda x: np.where(x['shape0.top_color'] > x['shape1.top_color'], 1, 0)),
        params=params)
    expect = [Grid([[1, 1, 0, 0], [1, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]]),
              Grid([[2, 2, 0, 0], [2, 2, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
              Grid([[3, 3, 0, 0], [3, 3, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])]

    state = create_test_state(x_shapes, y_shapes)
    state = state.update(has_layer=True)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out == expect


def test_expert():
    # 23
    params = GlobalParams()
    y_shapes = [[
        FilledRectangle(0, 4, 9, 1, 3),
        FilledRectangle(0, 6, 9, 1, 1),
        FilledRectangle(2, 0, 1, 9, 2)], [
        FilledRectangle(0, 1, 8, 1, 3),
        FilledRectangle(0, 4, 8, 1, 3),
        FilledRectangle(0, 6, 8, 1, 1),
        FilledRectangle(5, 0, 1, 10, 2)], [
        FilledRectangle(0, 1, 11, 1, 1),
        FilledRectangle(0, 3, 11, 1, 3),
        FilledRectangle(0, 6, 11, 1, 3),
        FilledRectangle(3, 0, 1, 10, 2),
        FilledRectangle(9, 0, 1, 10, 2)]]
    y = [Grid([[0, 0, 2, 0, 0, 0, 0, 0, 0],
               [0, 0, 2, 0, 0, 0, 0, 0, 0],
               [0, 0, 2, 0, 0, 0, 0, 0, 0],
               [0, 0, 2, 0, 0, 0, 0, 0, 0],
               [3, 3, 3, 3, 3, 3, 3, 3, 3],
               [0, 0, 2, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 2, 0, 0, 0, 0, 0, 0],
               [0, 0, 2, 0, 0, 0, 0, 0, 0]]),
         Grid([[0, 0, 0, 0, 0, 2, 0, 0],
               [3, 3, 3, 3, 3, 3, 3, 3],
               [0, 0, 0, 0, 0, 2, 0, 0],
               [0, 0, 0, 0, 0, 2, 0, 0],
               [3, 3, 3, 3, 3, 3, 3, 3],
               [0, 0, 0, 0, 0, 2, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 2, 0, 0],
               [0, 0, 0, 0, 0, 2, 0, 0],
               [0, 0, 0, 0, 0, 2, 0, 0]]),
         Grid([[0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0],
               [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
               [0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0],
               [0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0],
               [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
               [0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0],
               [0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0],
               [0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0]])]

    state = create_test_state(y_shapes, y_shapes)
    state = state.update(has_layer=True, y=y)
    manager = ArcManager(params)
    task_states = manager.decide(state)
    task, state = task_states[0]
    expert = DrawCanvasExpert(params)
    actions = expert.solve_problem(state, task)
    runtime_actions = actions[0].train_models(state, task)
    out_state = runtime_actions[0].perform(state, task)
    assert out_state.out == y
