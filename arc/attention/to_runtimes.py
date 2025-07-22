from .types import *
from ..graphic import *
from ..base import *
from .to_runtime.train_model import *
from .to_runtime.align_x import *
from .make_attention.find_shapes import *


def to_models(atn: TrainingAttention, output_train_shapes: list[list[Shape]],
              x_train: list[Grid], params: GlobalParams)->list[AttentionModel]:
    '''Generate models for predicting InferenceAttention.'''

    index_blob = gen_all_possible_index(output_train_shapes, atn.x_cluster_info)
    if index_blob is None:
        return []

    possible_sample_index, possible_x_index = index_blob
    df = create_df(x_train, output_train_shapes,
                   possible_sample_index, possible_x_index)
    has_syntactic = atn.syntactic_model is not None
    label = create_label(atn.sample_index, atn.x_index,
                         possible_sample_index, possible_x_index, has_syntactic)
    models = train_model(df, label, params)
    n_cols = len(atn.x_index[0])
    return [AttentionModel(model, n_cols, atn.x_cluster_info, atn.extra_shapes,
                           atn.syntactic_model) for model in models]


def to_runtimes(
        model: AttentionModel, output_test_shapes: list[list[Shape]],
        x_test: list[Grid])->Optional[InferenceAttention]:
    '''Predict InferenceAttention from the model.'''

    index_blob = gen_all_possible_index(output_test_shapes, model.x_cluster_info)
    if index_blob is None:
        return None

    sample_index, x_index = index_blob
    df = create_df(x_test, output_test_shapes, sample_index, x_index)
    correct_index = model.model.predict_bool(df)

    x_index2 = _add_syntactic(model.syntactic_model,
                              sample_index, x_index, output_test_shapes)
    result_sample_index, result_x_index = [], []
    for correct, sample_index, index in zip(correct_index, sample_index, x_index2):
        if correct:
            result_sample_index.append(sample_index)
            result_x_index.append(index)

    # need at least 1 row
    if len(result_sample_index) == 0:
        return None
    # the number of columns need to be the same
    if model.n_columns != len(result_x_index[0]):
        return None
    return InferenceAttention(result_sample_index, result_x_index, model.extra_shapes,
                              model.model, model.syntactic_model)


def _add_syntactic(
        model: Optional[MLModel], sample_index: list[int],
        x_index: list[list[int]], all_shapes: list[list[Shape]])->list[list[int]]:
    if model is None:
        return x_index
    prediction = predict_syntactic_shapes(model, sample_index, x_index, all_shapes)
    if prediction is None:
        return x_index
    return [index+[extra] for index, extra in zip(x_index, prediction)]
