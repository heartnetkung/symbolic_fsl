from ...arc.base import *
from ...arc.graphic import *
from ...arc.attention.low_level import *
from ...arc.attention.make_attention.find_shapes import *

shape1 = Unknown(0, 0, Grid([[-1, 1, -1], [1, 1, 1], [-1, 1, -1]]))
shape2 = FilledRectangle(1, 1, 1, 1, 1)
shape3 = FilledRectangle(1, 1, 1, 1, 2)
shape4 = FilledRectangle(2, 2, 2, 2, 2)
shape5 = FilledRectangle(3, 3, 3, 3, 3)
shape6 = FilledRectangle(6, 6, 6, 6, 6)


def test_common_y_shapes():
    result = find_common_y_shapes([[shape1, shape2, shape5], [shape1, shape3, shape4]])
    assert result == [shape1, shape2]


def test_common_y_shapes2():
    result = find_common_y_shapes([[shape1, shape2, shape5], [shape4, shape6]])
    assert result == []


def test_common_y_shapes3():
    problem_no = 11
    ds = read_datasets(DatasetChoice.train_v1)[problem_no]
    all_y_shapes = [list_sparse_objects(y_grid.remove_bg(), True)
                    for y_grid in ds.y_train]
    result = find_common_y_shapes(all_y_shapes)
    assert len(result) == 1
    assert result[0] == Unknown(0, 1, Grid([
        [2, -1, 7, -1, 2],
        [-1, 2, 7, 2, -1],
        [7, 7, 2, 7, 7],
        [-1, 2, 7, 2, -1],
        [2, -1, 7, -1, 2]]))


def test_syntactic_shapes():
    all_x_shapes = [[shape1, shape2], [shape3, shape1], [shape4, shape1]]
    sample_index = [0, 1, 2]
    x_index = [[1], [0], [0]]
    models = make_syntactic_models(sample_index, x_index,  all_x_shapes)
    assert len(models) == 5


def test_syntactic_shapes2():
    all_x_shapes = [[shape1, shape2], [shape3, shape1], [shape4]]
    sample_index = [0, 1, 2]
    x_index = [[1], [0], [0]]
    models = make_syntactic_models(sample_index, x_index,  all_x_shapes)
    assert len(models) == 0


def test_syntactic_shapes3():
    all_x_shapes = [[shape1, shape2], [shape3, shape1], [shape4, shape2]]
    sample_index = [0, 1, 2]
    x_index = [[1], [0], [0]]
    models = make_syntactic_models(sample_index, x_index,  all_x_shapes)
    assert len(models) == 0


def test_syntactic_shapes4():
    all_x_shapes = [[shape1, shape2], [shape3, shape1], [shape4, shape1, shape1]]
    sample_index = [0, 1, 2]
    x_index = [[1], [0], [0]]
    models = make_syntactic_models(sample_index, x_index,  all_x_shapes)
    assert len(models) == 0


def test_syntactic_prediction():
    model = ConstantColumnModel('shape0.top_color', 1)
    all_x_shapes = [[shape1, shape2], [shape3, shape1], [shape4, shape1]]
    sample_index = [0, 1, 2]
    x_index = [[1], [0], [0]]
    result = predict_syntactic_shapes(model, sample_index, x_index,  all_x_shapes)
    assert result is None  # one-colored shape in attention


def test_syntactic_prediction2():
    model = ConstantColumnModel('shape0.top_color', 1)
    all_x_shapes = [[shape2, shape5], [shape3, shape1], [shape4, shape1]]
    sample_index = [0, 1, 2]
    x_index = [[1], [0], [0]]
    result = predict_syntactic_shapes(model, sample_index, x_index,  all_x_shapes)
    assert result == [0, 1, 1]  # correct


def test_syntactic_prediction3():
    model = ConstantColumnModel('shape0.top_color', 1)
    all_x_shapes = [[shape2, shape5], [shape3, shape1], [shape4, shape1, shape2]]
    sample_index = [0, 1, 2]
    x_index = [[1], [0], [0]]
    result = predict_syntactic_shapes(model, sample_index, x_index,  all_x_shapes)
    assert result is None # two one-colored shapes in last sample
