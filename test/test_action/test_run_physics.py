from .util import *
from ...arc.graphic import draw_canvas


def test_basic():
    params = GlobalParams()
    x_shapes = [list_objects(x1.remove_bg())]
    y_shapes = [list_objects(y1.remove_bg())]
    action = RunPhysics(param=RunPhysicsParam.solid_south,
                        still_colors=set(), params=params)

    state = create_test_state(x_shapes, y_shapes).update(run_physics=False)
    program = AttentionExpertProgram(action, params)
    result = program.run(state)
    canvas = draw_canvas(y1.width, y1.height, result.out_shapes[0], 0)
    assert canvas == y1
    assert set(result.out_shapes[0]) == set(result.y_shapes[0])


x1 = Grid([
    [0, 2, 0, 4, 3],
    [5, 0, 0, 0, 0],
    [0, 0, 6, 0, 0],
    [5, 2, 0, 4, 0],
    [5, 0, 0, 0, 0]
])
y1 = Grid([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0],
    [5, 2, 0, 4, 0],
    [5, 2, 6, 4, 3]
])
