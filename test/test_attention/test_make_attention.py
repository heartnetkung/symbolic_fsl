from ...arc.base import *
from ...arc.graphic import *
from ...arc.attention.low_level import *
from ...arc.attention.make_attention.cluster_y import *
from ...arc.attention.make_attention.cluster_x import *
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