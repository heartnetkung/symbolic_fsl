from ...arc.ml import FunctionModel
import pandas as pd
import numpy as np


def test_fm():
    model = FunctionModel(lambda x:x['hello']+1)
    pred = model.predict(pd.DataFrame({'hello': [7, 92]}))
    assert np.array_equal(pred, [8, 93])
    assert "FunctionModel(lambda x:x['hello']+1))" == model.code
