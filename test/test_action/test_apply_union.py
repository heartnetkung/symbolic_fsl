from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[Unknown(0, 0, Grid([[1, -1, -1], [-1, 1, -1], [1, -1, -1]])),
                 Unknown(4, 0, Grid([[-1, 2, -1], [2, 2, 2], [-1, -1, -1]]))],

                [Unknown(0, 0, Grid([[1, 1, -1], [-1, -1, 1], [1, 1, -1]])),
                 Unknown(4, 0, Grid([[-1, 2, -1], [2, 2, 2], [-1, 2, -1]]))],

                [Unknown(0, 0, Grid([[-1, -1, 1], [1, 1, -1], [-1, 1, 1]])),
                 Unknown(4, 0, Grid([[-1, -1, -1], [2, -1, 2], [2, -1, 2]]))]]

    y_shapes = [
        [Unknown(0, 0, Grid([[1, 2, -1], [2, 2, 2], [1, -1, -1]]))],
        [Unknown(0, 0, Grid([[1, 2, -1], [2, 2, 2], [1, 2, -1]]))],
        [Unknown(0, 0, Grid([[-1, -1, 1], [2, 1, 2], [2, 1, 2]]))]
    ]
    action = ApplyUnion(feat_indexes=[0, 1])

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    print_pair(result)
    assert result.out_shapes == y_shapes
