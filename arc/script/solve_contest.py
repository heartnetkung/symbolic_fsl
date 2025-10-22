import os
from ..base import *
from .util import *
import json
from ..graphic import *
from ..manager import ArcManager
from ..expert import ArcRecruiter
import time

INPUT_FILE = os.path.join('/kaggle/input/arc-prize-2025',
                          'arc-agi_test_challenges.json')
OUTPUT_FILE = os.path.join('/kaggle/working', 'submission.json')
# INPUT_FILE = os.path.join(path.abspath(path.join(__file__, '../../../data/1.0')), 'arc-agi_training_challenges.json')
# OUTPUT_FILE = 'submission.json'
MAX_RETURN = 2
MAX_TIME_S = 300
MAX_TOTAL_TIME_S = 42000


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


def _solve_one(dataset: Dataset)->list:
    params = GlobalParams(
        enable_free_draw=False, enable_edge=False, enable_merge=False,
        enable_stack=False, enable_split=False)
    manager = ArcManager(params)
    hr = ArcRecruiter(params)
    result = _init_output(dataset)
    arc_result = solve_arc(dataset, manager, hr, params, max_time_s=MAX_TIME_S)
    for i in range(len(dataset.X_test)):
        for j, prediction in enumerate(arc_result.predictions[:MAX_RETURN]):
            result[i][f'attempt_{j+1}'] = prediction[i].data
    return result


def solve_contest():
    datasets = _read_dataset()
    output = {}
    end_time = time.time() + MAX_TOTAL_TIME_S

    for i, dataset in enumerate(datasets):
        print('solving', i)
        try:
            if time.time() > end_time:
                raise Exception('time_limit')
            output[dataset._id] = _solve_one(dataset)
            print('solved', dataset._id, output[dataset._id])
        except Exception as e:
            print(e)
            output[dataset._id] = _init_output(dataset)
    _write_result(output)


if __name__ == "__main__":
    solve_contest()
