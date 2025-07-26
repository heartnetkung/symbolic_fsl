from .visual_model import *
from .bound_scan import BoundScan, BoundScanModel
import numpy as np
import pandas as pd
from ...graphic import *
from ...constant import *
from ..model.model_factory import *



def bound_model_factory(grids: list[Grid], X: pd.DataFrame, y: np.ndarray,
                        params: GlobalParams)->list[VisualModel]:
    assert len(X) == len(y)
    assert y.shape[1] == 4

    result = []

    # dummy model
    label1, label2, label3, label4 = y.T
    models1 = regressor_factory(X, label1, params, 'bound_x')
    models2 = regressor_factory(X, label2, params, 'bound_y')
    models3 = regressor_factory(X, label3, params, 'bound_w')
    models4 = regressor_factory(X, label4, params, 'bound_h')
    result += [DummyVisualModel([model1, model2, model3, model4])
               for model1, model2, model3, model4 in model_selection(
        models1, models2, models3, models4)]

    # bound scan
    for rep in [BoundScan(True), BoundScan(False)]:
        X2 = rep.encode_feature(grids, X)
        y2 = rep.encode_label(grids, y)
        for model in classifier_factory(X2, y2, params, 'bound'):
            result.append(BoundScanModel(model, rep))
    return result
