from .visual_model import VisualModel
from .bound_scan import BoundScan, BoundScanModel
import numpy as np
import pandas as pd
from ...graphic import *
from ...constant import *
from ..model.model_factory import classifier_factory


def bound_model_factory(grids: list[Grid], X: pd.DataFrame, y: np.ndarray,
                        params: GlobalParams)->list[VisualModel]:
    assert len(X) == len(y)
    assert y.shape[1] == 4

    result = []
    for rep in [BoundScan(True), BoundScan(False)]:
        X2 = rep.encode_feature(grids, X)
        y2 = rep.encode_label(grids, y)
        for model in classifier_factory(X2, y2, params, 'bound'):
            result.append(BoundScanModel(model, rep))
    return result
