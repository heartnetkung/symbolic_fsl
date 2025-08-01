from .solve_one import *
from .solve_range import Report
from .params.previous_params import previous_v1_params
from .params.previous_params2 import previous_v1_params2
from .params.previous_params3 import previous_v1_params3
from .params.previous_params4 import previous_v1_params4
import traceback

SKIP = {31}


def solve_previous(index: int = -1)->None:
    report = Report()
    init_pandas()

    running_params = (previous_v1_params | previous_v1_params2 |
                      previous_v1_params3 | previous_v1_params4)
    if index != -1:
        try:
            running_params = {index: running_params[index]}
        except KeyError:
            raise Exception('incorrect index')

    for index, params in running_params.items():
        print(f'solving #{index}')
        if index in SKIP:
            print('skip')
            continue

        try:
            result = solve_one(index, DatasetChoice.train_v1, logging.ERROR, params)
            report.append(
                index, result.elapsed_time_s, result.correct == True,
                len(result.predictions), result.reasoning_result.path_count,
                result.planning_result.message, result.reasoning_result.message)
        except:
            print(traceback.format_exc())
    report.print()
    print('\a', file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) not in (1, 2):
        terminate(('incorrect usage! solve_previous.py [index]'))
    index = -1 if len(sys.argv) == 1 else int(sys.argv[1])
    solve_previous(index)
