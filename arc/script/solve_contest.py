import os
from ..base import *
from .util import *
import json
from ..graphic import *

INPUT_FILE = '/kaggle/input/arc-prize-2025/arc-agi_test-challenges.json'
OUTPUT_FILE = '/kaggle/working/submission.json'


def _read_dataset()->list[Dataset]:
    with open(INPUT_FILE) as f:
        challenges = json.load(f)
        result = []

        for key, challenge in challenges.items():
            X_train = [Grid(pair['input']) for pair in challenge['train']]
            y_train = [Grid(pair['output']) for pair in challenge['train']]
            X_test = [Grid(pair['input']) for pair in challenge['test']]
            result.append(Dataset(key, X_train, y_train, X_test, None))
        return result


def _write_result(result: dict[str, list])->None:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f)


def _init_output(dataset: Dataset)->list:
    result = []
    for _ in range(len(dataset.X_test)):
        result.append({'attempt_1': [[0]], 'attempt_2': [[0]]})
    return result


def _solve_one(dataset: Dataset):
    raise Exception('aa')


def solve_contest():
    datasets = _read_dataset()
    output = {}
    for dataset in datasets:
        try:
            output[dataset._id] = _solve_one(dataset)
        except Exception:
            output[dataset._id] = _init_output(dataset)
    _write_result(output)


if __name__ == "__main__":
    solve_contest()
