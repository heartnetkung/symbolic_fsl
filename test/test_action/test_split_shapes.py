from .util import *


def test_basic():
    params = GlobalParams()
    all_x_shapes = [
        [Unknown(2, 1, Grid([
            [5, 5, -1, -1, -1, -1],
            [5, 5, 5, 5, 5, -1],
            [-1, 5, 5, 5, -1, -1],
            [-1, 5, 5, 5, 5, 5],
            [-1, 5, -1, -1, 5, 5],
            [-1, -1, -1, 5, 5, 5]]))],
        [Unknown(1, 1, Grid([
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [-1, -1, 5, -1, -1, 5],
            [-1, -1, -1, 5, 5, 5],
            [-1, -1, -1, 5, 5, 5],
            [-1, -1, -1, 5, -1, -1]]))],
        [Unknown(1, 1, Grid([
            [5, 5, 5, 5, 5],
            [-1, -1, -1, 5, 5],
            [-1, -1, 5, -1, -1],
            [-1, -1, 5, 5, 5],
            [-1, -1, 5, 5, 5]]))]]

    all_y_shapes = [
        [FilledRectangle(2, 1, 2, 2, 8),
         FilledRectangle(4, 3, 2, 2, 8),
         FilledRectangle(6, 4, 2, 2, 8),
         FilledRectangle(4, 2, 3, 1, 2),
         FilledRectangle(5, 6, 3, 1, 2),
         FilledRectangle(3, 3, 1, 3, 2)],
        [FilledRectangle(1, 1, 2, 2, 8),
         FilledRectangle(4, 1, 2, 2, 8),
         FilledRectangle(5, 4, 2, 2, 8),
         FilledRectangle(3, 1, 1, 3, 2),
         FilledRectangle(6, 1, 1, 3, 2),
         FilledRectangle(4, 4, 1, 3, 2)],
        [FilledRectangle(4, 1, 2, 2, 8),
         FilledRectangle(4, 4, 2, 2, 8),
         FilledRectangle(3, 3, 1, 3, 2),
         FilledRectangle(1, 1, 3, 1, 2)]
    ]
    action = SplitShape(0)

    state = create_test_state(all_x_shapes, all_y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    for y_shapes, x_shapes in zip(all_y_shapes, result.out_shapes):
        assert set(x_shapes) == set(y_shapes)


def test_mixed():
    params = GlobalParams()
    all_x_shapes = [
        [Unknown(2, 1, Grid([
            [5, 5, -1, -1, -1, -1],
            [5, 5, 5, 5, 5, -1],
            [-1, 5, 5, 5, -1, -1],
            [-1, 5, 5, 5, 5, 5],
            [-1, 5, -1, -1, 5, 5],
            [-1, -1, -1, 5, 5, 5]]))],
        [Unknown(1, 1, Grid([
            [5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5],
            [-1, -1, 5, -1, -1, 5],
            [-1, -1, -1, 5, 5, 5],
            [-1, -1, -1, 5, 5, 5],
            [-1, -1, -1, 5, -1, -1]]))],
        [Unknown(1, 1, Grid([
            [5, 5, 5, 5, 5],
            [-1, -1, -1, 5, 5],
            [-1, -1, 5, -1, -1],
            [-1, -1, 5, 5, 5],
            [-1, -1, 5, 5, 5]]))]]

    all_y_shapes = [
        [FilledRectangle(2, 1, 2, 2, 8),
         Unknown(4, 3, Grid([[8, 8, -1, -1], [8, 8, 8, 8], [-1, -1, 8, 8]])),
         FilledRectangle(4, 2, 3, 1, 2),
         FilledRectangle(5, 6, 3, 1, 2),
         FilledRectangle(3, 3, 1, 3, 2)],
        [FilledRectangle(1, 1, 2, 2, 8),
         FilledRectangle(4, 1, 2, 2, 8),
         FilledRectangle(5, 4, 2, 2, 8),
         FilledRectangle(3, 1, 1, 3, 2),
         FilledRectangle(6, 1, 1, 3, 2),
         FilledRectangle(4, 4, 1, 3, 2)],
        [FilledRectangle(4, 1, 2, 2, 8),
         FilledRectangle(4, 4, 2, 2, 8),
         FilledRectangle(3, 3, 1, 3, 2),
         FilledRectangle(1, 1, 3, 1, 2)]
    ]
    action = SplitShape(0)

    state = create_test_state(all_x_shapes, all_y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    for i, (y_shapes, x_shapes) in enumerate(zip(all_y_shapes, result.out_shapes)):
        if i == 0:
            set(x_shapes) == {FilledRectangle(2, 1, 2, 2, 8),
                              FilledRectangle(4, 3, 2, 2, 8),
                              FilledRectangle(6, 4, 2, 2, 8),
                              FilledRectangle(4, 2, 3, 1, 2),
                              FilledRectangle(5, 6, 3, 1, 2),
                              FilledRectangle(3, 3, 1, 3, 2)}
        else:
            assert set(x_shapes) == set(y_shapes)
