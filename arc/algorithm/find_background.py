from __future__ import annotations
from ..graphic import *
from ..base import *
from copy import deepcopy
import networkx as nx
from scipy.stats import mode
from dataclasses import dataclass
from typing import Union
from ..ml import MLModel, ConstantModel, ColumnModel
import pandas as pd

MISSING_VALUE = 10
COMMON_BG = 0


def find_backgrounds(state: ArcTrainingState)->list[tuple[MLModel, MLModel]]:
    '''
    Find the possible choices of background for all grids in the dataset
    (including y_test).
    '''
    X_train_top_colors = [grid.get_top_color() for grid in state.x]
    y_train_top_colors = [grid.get_top_color() for grid in state.y]
    X_train_bg = [find_background(grid) for grid in state.x]
    y_train_bg = [find_background(grid) for grid in state.y]
    x_common_colors = _find_common_colors(state.x)
    y_common_colors = _find_common_colors(state.y)
    X_global_top_color = _find_global_top_color(state.x)
    y_global_top_color = _find_global_top_color(state.y)

    result = []
    if COMMON_BG in x_common_colors:
        result.append((ConstantModel(COMMON_BG), ConstantModel(COMMON_BG)))

    if len(set(X_train_top_colors)) > 1:
        x_dynamic = True
        for x_grid, y_grid in zip(state.x, state.y):
            x_colors = x_grid.color_count
            x_top_color = x_grid.get_top_color()
            if x_top_color not in y_grid.list_colors():
                x_dynamic = False
            if x_colors[x_top_color]/x_grid.width/x_grid.height < 0.4:
                x_dynamic = False
        if x_dynamic:
            result.append((ColumnModel('x_top_color'),
                           ColumnModel('x_top_color')))
            if len(X_train_top_colors) > len(set(X_train_top_colors)):
                dynamic_mode = mode(X_train_top_colors).mode
                if dynamic_mode in x_common_colors:
                    result.append((ConstantModel(dynamic_mode),
                               ConstantModel(dynamic_mode)))

    if ((X_global_top_color in x_common_colors) and
        (y_global_top_color in y_common_colors) and
            (X_global_top_color != y_global_top_color)):
        result.append((ConstantModel(X_global_top_color),
                       ConstantModel(y_global_top_color)))

    x_mode = mode(X_train_bg+X_train_top_colors).mode
    y_mode = mode(y_train_bg+y_train_top_colors).mode
    if x_mode == y_mode:
        if x_mode in x_common_colors and x_mode != COMMON_BG:
            result.append((ConstantModel(x_mode), ConstantModel(x_mode)))
    if len(result) > 0:
        return result
    return [(ConstantModel(COMMON_BG), ConstantModel(COMMON_BG))]  # type:ignore


def make_background_df(state: ArcState)->pd.DataFrame:
    data = {'x_top_color': [grid.get_top_color() for grid in state.x]}
    return pd.DataFrame(data)


def find_background(grid: Grid)->int:
    diagonal = True
    grid2, color_map = _draw_color_map(grid, diagonal)
    return _find_most_touch(grid2, diagonal, color_map)


def _find_common_colors(grids: list[Grid])->set[int]:
    result: Optional[set[int]] = None
    for grid in grids:
        if result == None:
            result = grid.list_colors()
        else:
            result &= grid.list_colors()
    return result if result is not None else set()


def _find_global_top_color(grids: list[Grid])->int:
    data: dict[int, int] = {}
    for grid in grids:
        for row in grid.data:
            for cell in row:
                data[cell] = data.get(cell, 0)+1
    return max(data.keys(), key=data.__getitem__)


def _find_most_touch(grid: Grid, diagonal: bool, color_map: dict[int, int])->int:
    edges, size_map = [], {}
    for j in range(grid.width):
        for i in range(grid.height):
            edges += _list_surrounding_edges(grid, i, j, diagonal)
            current_value = grid.data[i][j]
            size_map[current_value] = size_map.get(current_value, 0)+1

    if len(edges) == 0:
        return grid.data[0][0]

    g = nx.Graph()
    g.add_edges_from(edges)
    all_nodes = [(node, len(adjacency)*100+size_map[node])
                 for node, adjacency in g.adjacency()]
    most_touch_node = max(all_nodes, key=lambda x: x[1])[0]
    return color_map[most_touch_node]


def _list_surrounding_edges(grid: Grid, i: int, j: int, diagonal: bool)->list[int]:
    edges = []
    current_value = grid.data[i][j]

    adjacent_value = _safe_access(grid, j-1, i)
    if adjacent_value != MISSING_VALUE:
        edges.append((current_value, adjacent_value))

    adjacent_value = _safe_access(grid, j+1, i)
    if adjacent_value != MISSING_VALUE:
        edges.append((current_value, adjacent_value))

    adjacent_value = _safe_access(grid, j, i+1)
    if adjacent_value != MISSING_VALUE:
        edges.append((current_value, adjacent_value))

    adjacent_value = _safe_access(grid, j, i-1)
    if adjacent_value != MISSING_VALUE:
        edges.append((current_value, adjacent_value))

    if diagonal:
        adjacent_value = _safe_access(grid, j+1, i+1)
        if adjacent_value != MISSING_VALUE:
            edges.append((current_value, adjacent_value))

        adjacent_value = _safe_access(grid, j+1, i-1)
        if adjacent_value != MISSING_VALUE:
            edges.append((current_value, adjacent_value))

        adjacent_value = _safe_access(grid, j-1, i+1)
        if adjacent_value != MISSING_VALUE:
            edges.append((current_value, adjacent_value))

        adjacent_value = _safe_access(grid, j-1, i-1)
        if adjacent_value != MISSING_VALUE:
            edges.append((current_value, adjacent_value))

    return edges


def _safe_access(grid: Grid, x: int, y: int)->int:
    if x < 0 or y < 0:
        return MISSING_VALUE
    if x >= grid.width or y >= grid.height:
        return MISSING_VALUE
    return grid.data[y][x]


def _draw_color_map(grid: Grid, diagonal: bool)->tuple[Grid, dict[int, int]]:
    grid2 = deepcopy(grid)
    current_obj = -2
    color_map = {}

    for j in range(grid.width):
        for i in range(grid.height):
            if grid2.data[i][j] < 0:
                continue
            _expand(grid2, i, j, current_obj, grid.data[i][j], diagonal)
            color_map[current_obj] = grid.data[i][j]
            current_obj -= 1
    return grid2, color_map


def _expand(grid: Grid, i: int, j: int, current_obj: int,
            color: int, diagonal: bool):
    if i < 0 or j < 0 or i >= grid.height or j >= grid.width:
        return
    if grid.data[i][j] < 0:
        return
    if grid.data[i][j] != color:
        return
    grid.data[i][j] = current_obj
    _expand(grid, i+1, j, current_obj, color, diagonal)
    _expand(grid, i-1, j, current_obj, color, diagonal)
    _expand(grid, i, j+1, current_obj, color, diagonal)
    _expand(grid, i, j-1, current_obj, color, diagonal)
    if diagonal:
        _expand(grid, i+1, j+1, current_obj, color, diagonal)
        _expand(grid, i+1, j-1, current_obj, color, diagonal)
        _expand(grid, i-1, j+1, current_obj, color, diagonal)
        _expand(grid, i-1, j-1, current_obj, color, diagonal)
