from .visual_model import *
from .bound_scan import BoundScan, BoundScanModel
import numpy as np
import pandas as pd
from ...graphic import *
from ...constant import *
from ..model.model_factory import *


def bound_model_factory(grids: list[Grid], X: pd.DataFrame, y: np.ndarray,
                        params: GlobalParams)->list[VisualModel]:
    assert len(X) == len(y) == len(grids)
    assert y.shape[1] == 4

    result = []

    # dummy model
    label1, label2, label3, label4 = y.T
    labels = [label1, label2, label3, label4]
    label_types = [LabelType.reg]*4
    all_models = make_all_models(X, params, 'bound', labels, label_types)
    result += [DummyVisualModel([model1, model2, model3, model4])
               for model1, model2, model3, model4 in model_selection(*all_models)]

    # bound scan
    for inclusive in BOOLS:
        rep = BoundScan(inclusive)
        X2 = rep.encode_feature(grids, X)
        y2 = rep.encode_label(grids, y)
        for model in make_classifier(X2, y2, params, f'bound_scan_{inclusive}'):
            result.append(BoundScanModel(model, rep))
    return result
