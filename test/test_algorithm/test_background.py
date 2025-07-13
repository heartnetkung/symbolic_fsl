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
        training_state = ds[i].to_training_state()
        result = find_backgrounds(training_state)
        df_train = make_background_df(training_state)
        df_test = make_background_df(ds[i].to_inference_state())
        found = False

        for x_model, y_model in result:
            if (BG[i]['X_train'] == x_model.predict_int(df_train) and
                BG[i]['y_train'] == y_model.predict_int(df_train) and
                BG[i]['X_test'] == x_model.predict_int(df_test) and
                    BG[i]['y_test'] == y_model.predict_int(df_test)):
                found = True
        assert found
