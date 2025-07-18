from .util import *


def test_basic():
    params = GlobalParams()
    x_shapes = [[FilledRectangle(0, 0, 1, 1, 1), FilledRectangle(0, 2, 1, 1, 2),
                 FilledRectangle(7, 7, 1, 1, 1)]]
    x = [make_grid(8, 8)]
    bg = [0]

    state = ArcTrainingState(
        x, x, None, bg, bg, False, 5, x_shapes, x_shapes, x_shapes,
        True, True, True, True)

    action = MergeNearby(MergeNearbyParam.normal)
    out_state = action.perform_train(state, ModelFreeTask())
    assert out_state.out_shapes == [[
        Unknown(0, 0, Grid([[1], [-1], [2]])),
        FilledRectangle(7, 7, 1, 1, 1)]]
