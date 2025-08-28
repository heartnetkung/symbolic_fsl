from .util import *
from ..base import *


def print_grid(index: int, choice: DatasetChoice)->None:
    ds = read_datasets(choice)[index]
    for i, (x_grid, y_grid) in enumerate(zip(ds.X_train, ds.y_train)):
        print(f'x{index}_{i} = ', end='')
        x_grid.print_grid2()
        print(f'y{index}_{i} = ', end='')
        y_grid.print_grid2()


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        terminate(('incorrect usage! print_grid.py <index>'
                   ' [train_v1|eval_v1|train_v2|eval_v2]'))

    choice_str = sys.argv[2] if len(sys.argv) >= 3 else None
    dataset_choice = parse_choice(choice_str)
    print_grid(int(sys.argv[1]), dataset_choice)
