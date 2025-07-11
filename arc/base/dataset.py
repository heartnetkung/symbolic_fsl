from __future__ import annotations
from ..graphic import Grid
import json
from dataclasses import dataclass
from typing import Optional
from os import path
from functools import cached_property,cache
from enum import Enum
from .arc_state import ArcState

INPUT_FOLDER = path.abspath(path.join(__file__, '../../../data/'))


class DatasetChoice(Enum):
    train_v1 = 0
    eval_v1 = 1
    train_v2 = 2
    eval_v2 = 3

    def get_folder(self)->str:
        if (self == DatasetChoice.train_v1 or
                self == DatasetChoice.eval_v1):
            return '1.0'
        return '2.0'

    def get_challenge_filename(self)->str:
        if (self == DatasetChoice.train_v1 or
                self == DatasetChoice.train_v2):
            return 'arc-agi_training_challenges.json'
        return 'arc-agi_evaluation_challenges.json'

    def get_solution_filename(self)->str:
        if (self == DatasetChoice.train_v1 or
                self == DatasetChoice.train_v2):
            return 'arc-agi_training_solutions.json'
        return 'arc-agi_evaluation_solutions.json'


@dataclass(frozen=True)
class Dataset:
    _id: str
    X_train: list[Grid]
    y_train: list[Grid]
    X_test: list[Grid]
    y_test: Optional[list[Grid]]

    @cached_property
    def all_x(self)->list[Grid]:
        return self.X_train+self.X_test

    @cached_property
    def all_y(self)->list[Grid]:
        assert self.y_test is not None
        return self.y_train+self.y_test

    def to_initial_state(self)->ArcState:
        return ArcState(self.X_train, self.X_test, self.y_train, self.y_test)


def _get_json(filename: str, version: str) -> dict:
    with open(path.join(INPUT_FOLDER, version, filename)) as f:
        d = json.load(f)
        return d


@cache
def read_datasets(choice: DatasetChoice)->dict[int, Dataset]:
    return _read_datasets(choice.get_challenge_filename(),
                          choice.get_solution_filename(),
                          choice.get_folder())


def _read_datasets(challenge_file: str, solution_file: str,
                   version: str)->dict[int, Dataset]:
    challenges = _get_json(challenge_file, version)
    solutions = _get_json(solution_file, version)

    all_dataset = {}
    for i, key in enumerate(challenges.keys()):
        challenge = challenges[key]
        X_train = [Grid(pair['input']) for pair in challenge['train']]
        y_train = [Grid(pair['output']) for pair in challenge['train']]
        X_test = [Grid(pair['input']) for pair in challenge['test']]
        y_test = [Grid(grid) for grid in solutions[key]]
        all_dataset[i] = Dataset(key, X_train, y_train, X_test, y_test)
    return all_dataset
