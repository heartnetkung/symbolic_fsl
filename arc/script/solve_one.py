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


def init(log_level: int)->GlobalParams:
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    warnings.simplefilter("ignore", category=UserWarning)
    logging.basicConfig(stream=sys.stdout, format='%(message)s')
    logging.getLogger('arc.base.plan').setLevel(log_level)
    logging.getLogger('arc.base.reason').setLevel(log_level)
    logging.getLogger('arc.base.solve_arc').setLevel(log_level)
    logging.getLogger('arc.ml.model.model_factory').setLevel(log_level)
    init_pandas()
    return GlobalParams()


def solve_one(index: int, choice: DatasetChoice, log_level: int,
              params: Optional[GlobalParams] = None)->ArcResult:
    if params is None:
        params = init(log_level)
    dataset = read_datasets(choice)[index]
    start = time.time()

    manager = ArcManager(params)
    hr = ArcRecruiter(params)
    result = solve_arc(dataset, manager, hr, params)

    print(result)

    if (result.correct_trace is not None) and (
            log_level in (logging.INFO, logging.DEBUG)):
        print('\ncorrect trace: ', result.correct_trace)
    print('\nelapsed time:', time.time()-start)
    print('\a', file=sys.stderr)
    return result


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3, 4):
        terminate(('incorrect usage! solve_one.py <index>'
                   ' [train_v1|eval_v1|train_v2|eval_v2] [error|debug|info]'))

    choice_str = sys.argv[2] if len(sys.argv) >= 3 else None
    dataset_choice = parse_choice(choice_str)
    log_str = sys.argv[3] if len(sys.argv) >= 4 else None
    log_level = parse_log(log_str)
    solve_one(int(sys.argv[1]), dataset_choice, log_level)
