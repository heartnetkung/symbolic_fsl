from .solve_one import *
import traceback

SKIP = {0, 4, 8, 57, 58, 59}


class Report:
    def __init__(self)->None:
        self.indexes = []
        self.data = {'duration': [], 'success': [], 'n_solution': [], 'path_count': [],
                     'plan_message': [], 'reason_message': []}

    def append(self, index: int, duration: float, success: bool, n_solution: int,
               path_count: int, plan_message: str, reason_message: str)->None:
        self.indexes.append(index)
        self.data['duration'].append(duration)
        self.data['success'].append(success)
        self.data['n_solution'].append(n_solution)
        self.data['path_count'].append(path_count)
        self.data['plan_message'].append(plan_message)
        self.data['reason_message'].append(reason_message)

    def print(self)->None:
        df = pd.DataFrame(self.data, index=self.indexes)  # type:ignore
        print()
        print(df.sort_index())
        print(f'\nsuccess: {df["success"].sum()}/{len(df)}')
        print("average time: {:.1f}".format(df["duration"].mean()))
        print("max time: {:.1f}".format(df["duration"].max()))


def solve_range(start: int, end: int, choice: DatasetChoice)->None:
    report = Report()
    for i in range(start, end):
        if i in SKIP:
            continue

        print(f'solving #{i}')
        try:
            result = solve_one(i, choice, logging.ERROR)
            report.append(
                i, result.elapsed_time_s, result.correct == True,
                len(result.predictions), result.reasoning_result.path_count,
                result.planning_result.message, result.reasoning_result.message)
        except:
            print(traceback.format_exc())
    report.print()
    print('\a', file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        terminate(('incorrect usage! solve_range.py <start> <end>'
                   ' [train_v1|eval_v1|train_v2|eval_v2]'))

    choice_str = sys.argv[3] if len(sys.argv) >= 4 else None
    dataset_choice = parse_choice(choice_str)
    solve_range(int(sys.argv[1]), int(sys.argv[2]), dataset_choice)
