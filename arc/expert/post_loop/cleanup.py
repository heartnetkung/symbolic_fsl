from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
import pandas as pd
from ..util import *


class CleanUp(ModelBasedArcAction[CleanUpTask, CleanUpTask]):
    def __init__(self, filter_model: MLModel, params: GlobalParams)->None:
        self.filter_model = filter_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: CleanUpTask)->Optional[ArcState]:
        assert state.out_shapes != None

        df = make_single_shape_df(state)
        is_selected = self.filter_model.predict_bool(df)
        all_new_shapes, offset = [], 0

        for shapes in state.out_shapes:
            row = [shape for include, shape in zip(is_selected, shapes) if include]
            all_new_shapes.append(row)
            offset += len(row)
        return state.update(out_shapes=all_new_shapes)

    def train_models(self, state: ArcTrainingState,
                     task: CleanUpTask)->list[InferenceAction]:
        assert isinstance(self.filter_model, MemorizedModel)

        df = make_single_shape_df(state)
        models = classifier_factory(df, self.filter_model.result, self.params, 'cleanup')
        return [CleanUp(model, self.params) for model in models]
