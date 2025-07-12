from ...arc.base import *
from ...arc.graphic import *
from ...arc.algorithm.find_background import *
from .bg_fixture import BG

# this is the fixture generation code
# def test_basic():
#     ds = read_datasets(DatasetChoice.train_v2)
#     for i in range(100):
#         result = find_backgrounds(ds[i])
#         if len(result) == 1:
#             print(result[0], f',#{i}')
#         else:
#             print(result[0], f',#{i} {result[1:]}')
#     assert False


def test_fixture():
    ds = read_datasets(DatasetChoice.train_v2)
    for i in range(100):
        result = find_backgrounds(ds[i])
        assert BG[i] in result
