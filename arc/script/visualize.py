import pandas as pd
import sys
from .util import *
from termcolor import colored
import itertools
import numpy as np
from typing import Optional
from ..base import *
from ..graphic import Grid

CELL_INDENT = 31
COLORS = ['grey', 'red', 'green', 'yellow', 'blue',
          'magenta', 'cyan', 'white', 'magenta', 'red']


def make_color_map(grids: list[Grid])->dict[int, int]:
    '''
    Terminal only supports 8 background color but we have 10 color input.
    '''
    all_elements = []
    for grid in grids:
        all_elements += list(itertools.chain(*grid.data))
    el, count = np.unique(all_elements, return_counts=True)
    ranking = pd.DataFrame({'el': el, 'count': count}).join(
        pd.DataFrame({'el': range(10)}), how='right', lsuffix='_', on='el').sort_values(
        'count', ascending=False, ignore_index=True)
    assert ranking is not None

    ranking_el = ranking['el']
    assert ranking_el is not None

    ans = {}
    for i in range(len(ranking_el)):
        ans[ranking_el[i]] = i
    return ans


def draw_cell(cell: int, color_map: dict)->None:
    print(colored(str(cell), COLORS[color_map[cell]], attrs=["reverse"]), end='')


def draw_grid(index: int, grid1: Grid, grid2: Grid, color_map: dict,
              label1: str, label2: str)->None:
    height1, width1 = grid1.height, grid1.width
    height2, width2 = grid2.height, grid2.width

    # draw header
    label1 = '{}) {} {}x{}'.format(index, label1, width1, height1)
    label2 = '{} {}x{}'.format(label2, width2, height2)
    print('\n ' + label1+(' '*(CELL_INDENT-len(label1))) +
          label2+(' '*(CELL_INDENT-len(label2))))

    # draw body
    for line in range(0, max(height1, height2)):
        print(' ', end='')
        if line < height1:
            for cell in grid1.data[line]:
                draw_cell(cell, color_map)
            print(' '*(CELL_INDENT-width1), end='')
        else:
            print(' '*CELL_INDENT, end='')
        if line < height2:
            for cell in grid2.data[line]:
                draw_cell(cell, color_map)
        print('')


def visualize(index: int, choice: DatasetChoice, train_only: bool)->None:
    dataset = read_datasets(choice)[index]
    color_map = make_color_map(dataset.X_train)
    if choice == DatasetChoice.train_v1:
        print('v2_index:', lookup_v2_problem_no(index))
    print('training_samples:', len(dataset.X_train))
    if not train_only:
        print('test_samples:', len(dataset.X_test))

    if train_only:
        all_data = zip(dataset.X_train, dataset.y_train)
    else:
        all_data = zip(dataset.all_x, dataset.all_y)

    pair_num = 0
    for x_grid, y_grid in all_data:
        # x_grid.print_grid2()
        # y_grid.print_grid2()
        draw_grid(pair_num, x_grid, y_grid, color_map, 'Input', 'Output')
        pair_num += 1
    print()


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3, 4):
        terminate(('incorrect usage! visualize.py <index>'
                   ' [train_v1|eval_v1|train_v2|eval_v2] [train_only]'))

    choice_str = sys.argv[2] if len(sys.argv) >= 3 else None
    dataset_choice = parse_choice(choice_str)
    train_only = True if len(sys.argv) >= 4 and sys.argv[3] == 'train_only' else False
    visualize(int(sys.argv[1]), dataset_choice, train_only)
