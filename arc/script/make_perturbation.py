from __future__ import annotations
from ..base import *
from ..graphic import *
from .util import *
from collections.abc import Callable


def make_perturbation():
    ds = read_datasets(DatasetChoice.train_v1)
    write_datasets(_perturb(ds, _color_shift), DatasetChoice.train_v1_color_shift)
    write_datasets(_perturb(ds, _fliph), DatasetChoice.train_v1_fliph)
    write_datasets(_perturb(ds, _flipv), DatasetChoice.train_v1_flipv)
    write_datasets(_perturb(ds, _transpose), DatasetChoice.train_v1_transpose)


def _color_shift(grid: Grid)->Grid:
    mapping = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 1}
    return Grid([[mapping[grid.data[i][j]] for j in range(grid.width)]
                 for i in range(grid.height)])


def _fliph(grid: Grid)->Grid:
    return grid.flip_h()


def _flipv(grid: Grid)->Grid:
    return grid.flip_v()


def _transpose(grid: Grid)->Grid:
    return grid.transpose()


def _perturb(datasets: dict[int, Dataset],
             func: Callable[[Grid], Grid])->dict[int, Dataset]:
    result = {}
    for key, dataset in datasets.items():
        assert dataset.y_test is not None
        result[key] = Dataset(
            dataset._id,
            [func(grid) for grid in dataset.X_train],
            [func(grid) for grid in dataset.y_train],
            [func(grid) for grid in dataset.X_test],
            [func(grid) for grid in dataset.y_test])
    return result


if __name__ == "__main__":
    make_perturbation()
