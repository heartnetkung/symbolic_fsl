from ...arc.base import *
from ...arc.graphic import *
from ...arc.attention.low_level import *
from ...arc.attention.make_attention.cluster_y import *
from ...arc.attention.make_attention.cluster_x import *
from ...arc.attention import *
import pandas as pd


def test_cluster_y():
    all_x_shapes = [[FilledRectangle(0, 0, 1, 1, 1), FilledRectangle(0, 0, 1, 3, 2)],
                    [FilledRectangle(0, 0, 2, 2, 1)]]
    all_y_shapes = [[FilledRectangle(0, 0, 1, 1, 2)], [FilledRectangle(0, 0, 2, 2, 2)]]
    expect = {'sample_id': [0, 1],
              'y_index': [0, 0],
              'y_label': [0, 0]}
    rel_df = gen_rel_product(all_x_shapes, all_y_shapes)
    possible_clusters = cluster_y(rel_df)
    assert len(possible_clusters) == 1
    to_check = possible_clusters[0][['sample_id', 'y_index', 'y_label']]
    assert pd.DataFrame(expect).equals(to_check)


def test_cluster_y2():
    all_x_shapes = [[FilledRectangle(0, 0, 1, 1, 1), FilledRectangle(0, 0, 1, 3, 2)],
                    [FilledRectangle(0, 0, 2, 2, 1)]]
    all_y_shapes = [[FilledRectangle(0, 0, 1, 1, 2), FilledRectangle(0, 0, 1, 1, 1)],
                    [FilledRectangle(0, 0, 2, 2, 1), FilledRectangle(0, 0, 2, 2, 2)]]
    expect = {'sample_id': [0, 1],
              'y_index': [0, 1],
              'y_label': [0, 0]}
    rel_df = gen_rel_product(all_x_shapes, all_y_shapes)
    possible_clusters = cluster_y(rel_df)
    assert len(possible_clusters) == 1
    to_check = possible_clusters[0][['sample_id', 'y_index', 'y_label']]
    assert pd.DataFrame(expect).equals(to_check)


def test_cluster_x():
    all_x_shapes = [[FilledRectangle(0, 0, 1, 1, 1), FilledRectangle(0, 0, 1, 3, 2)],
                    [FilledRectangle(0, 0, 2, 2, 1)]]
    all_y_shapes = [[FilledRectangle(0, 0, 1, 1, 2)], [FilledRectangle(0, 0, 2, 2, 2)]]
    expect = {'sample_id': [0, 1],
              'y_index': [0, 0],
              'x_index': [0, 0],
              'y_label': [0, 0],
              'x_label': [0, 0]}
    rel_df = gen_rel_product(all_x_shapes, all_y_shapes)
    possible_y_clusters = cluster_y(rel_df)
    assert len(possible_y_clusters) == 1
    possible_x_clusters = cluster_x(rel_df, possible_y_clusters[0])
    assert len(possible_x_clusters) == 1
    to_check = possible_x_clusters[0][list(expect.keys())]
    assert pd.DataFrame(expect).equals(to_check)


def test_predict():
    all_x_shapes = [[FilledRectangle(0, 0, 1, 1, 1), FilledRectangle(0, 0, 1, 3, 2)],
                    [FilledRectangle(0, 0, 2, 2, 1)]]
    all_y_shapes = [[FilledRectangle(0, 0, 1, 1, 2)], [FilledRectangle(0, 0, 2, 2, 2)]]
    all_x_test_shapes = [
        [FilledRectangle(1, 2, 3, 4, 1), FilledRectangle(5, 6, 7, 8, 2)]]
    params = GlobalParams()
    grid = Grid([[1, 2], [3, 4]])
    x_train_grids, x_test_grids = [grid, grid], [grid]

    attentions = make_attentions(all_x_shapes, all_y_shapes, x_train_grids)
    assert len(attentions) == 2
    assert attentions[0].sample_index == [0, 1]
    assert attentions[0].x_index == [[0], [0]]
    assert attentions[0].y_index == [0, 0]
    assert attentions[0].x_cluster_info == [1]

    models = to_models(
        attentions[0], all_x_shapes, x_train_grids, all_x_shapes, params)
    assert len(models) > 3

    attention2 = to_runtimes(
        models[0], all_x_test_shapes, x_test_grids, all_x_test_shapes)
    assert attention2.sample_index == [0]
    assert attention2.x_index == [[0]]
