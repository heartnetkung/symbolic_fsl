import sys
from os import path
import pickle
from typing import Any, Optional
from ..base import *
import logging
import pandas as pd

FILE_PATH = path.abspath(path.join(__file__, '../../../'))


def terminate(msg: str) -> None:
    print(msg, file=sys.stderr)
    exit(1)


def write_obj(filename: str, obj: Any)->None:
    with open(path.join(FILE_PATH, filename), 'wb') as f:
        pickle.dump(obj, f)


def read_obj(filename: str)->Any:
    with open(path.join(FILE_PATH, filename), 'rb') as f:
        return pickle.load(f)


def parse_choice(choice: Optional[str] = None)->DatasetChoice:
    result = DatasetChoice.train_v1
    if choice is not None:
        try:
            result = DatasetChoice[choice]
        except KeyError:
            terminate('valid choices are train_v1|test_v1|train_v2|test_v2')
    return result


def parse_log(choice: Optional[str] = None)->int:
    result = logging.INFO
    mapping = {'error': logging.ERROR, 'debug': logging.DEBUG, 'info': logging.INFO}

    if choice is not None:
        try:
            result = mapping[choice]
        except KeyError:
            terminate('valid choices are error|debug|info')
    return result


def lookup_v2_problem_no(v1_problem_no: int)->int:
    try:
        file_path = path.join(FILE_PATH, 'data/dataset_version_mapping.csv')
        df = pd.read_csv(file_path)
        result = df[df['v1_index'] == v1_problem_no][['v2_index']]
        assert isinstance(result, pd.DataFrame)
        return int(result.iloc[0, 0])
    except:
        return -1
