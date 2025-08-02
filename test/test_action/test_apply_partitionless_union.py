from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[Unknown(0, 0, Grid([
        [-1, 3, -1, 3],
        [3, 3, 3, -1],
        [-1, -1, -1, 3],
        [3, 3, 3, -1],
        [-1, -1, 1, 1],
        [-1, -1, 1, 1],
        [-1, 1, -1, -1],
        [1, 1, -1, -1]
    ]))]]

    y_shapes = [
        [Unknown(0, 0, Grid([
            [-1, 3, 1, 3],
            [3, 3, 3, 1],
            [-1, 1, -1, 3],
            [3, 3, 3, -1]
        ]))],
    ]

    action = ApplyPartitionlessUnion(
        order=[1, 0], row_count=2, col_count=1, params=params)

    state = create_test_state(x_shapes, y_shapes).update(partitionless_logic=False)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    print_pair(result)
    assert result.out_shapes == y_shapes
