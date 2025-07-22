from ..base import *
from ..graphic import *
from .util import *
from ..manager import ArcManager
import logging
import re
import warnings
import time
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
from ..expert.recruiter import ArcRecruiter

MAX_TIME_S = 600
MAX_PLAN_DEPTH = 15
MAX_PLAN_ITR = 5000


def init(log_level: int)->GlobalParams:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(stream=sys.stdout, format='%(message)s')
    logging.getLogger('arc.base.plan').setLevel(log_level)
    logging.getLogger('arc.base.reason').setLevel(log_level)
    logging.getLogger('arc.base.solve_arc').setLevel(log_level)
    logging.getLogger('arc.ml.model.model_factory').setLevel(log_level)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    return GlobalParams()


def plan_one(index: int, choice: DatasetChoice, log_level: int)->None:
    params = init(logging.ERROR)
    dataset = read_datasets(choice)[index]
    start = time.time()

    manager = ArcManager(params)
    hr = ArcRecruiter(params)
    criteria = ArcSuccessCriteria()
    result = plan(dataset.to_training_state(), manager, hr,
                  criteria, MAX_PLAN_DEPTH, MAX_PLAN_ITR, MAX_TIME_S)

    result.plan.trim()
    print(result.plan)
    print('\nmessage:',result.message)
    print('elapsed time:', time.time()-start)
    print('\a', file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3, 4):
        terminate(('incorrect usage! plan_one.py <index>'
                   ' [train_v1|eval_v1|train_v2|eval_v2] [error|debug|info]'))

    choice_str = sys.argv[2] if len(sys.argv) >= 3 else None
    dataset_choice = parse_choice(choice_str)
    log_str = sys.argv[3] if len(sys.argv) >= 4 else None
    log_level = parse_log(log_str)
    plan_one(int(sys.argv[1]), dataset_choice, log_level)
