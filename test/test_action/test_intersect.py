from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[Unknown(0, 0, Grid([[1, -1, -1], [-1, 1, -1], [1, -1, -1]])),
                 Unknown(4, 0, Grid([[-1, 1, -1], [1, 1, 1], [-1, -1, -1]]))],
                [Unknown(0, 0, Grid([[1, 1, -1], [-1, -1, 1], [1, 1, -1]])),
                 Unknown(4, 0, Grid([[-1, 1, -1], [1, 1, 1], [-1, 1, -1]]))],
                [Unknown(0, 0, Grid([[-1, -1, 1], [1, 1, -1], [-1, 1, 1]])),
                 Unknown(4, 0, Grid([[-1, -1, -1], [1, -1, 1], [1, -1, 1]]))]]

    y_shapes = [
        [Unknown(0, 0, Grid([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]))],
        [Unknown(0, 0, Grid([[-1, 1, -1], [-1, -1, 1], [-1, 1, -1]]))],
        [Unknown(0, 0, Grid([[-1, -1, -1], [1, -1, -1], [-1, -1, 1]]))]
    ]
    action = ApplyLogic(feat_indexes=[0, 1], color=1, type=LogicType.and_)

    state = create_test_state(x_shapes, y_shapes)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    assert result.out_shapes == y_shapes
