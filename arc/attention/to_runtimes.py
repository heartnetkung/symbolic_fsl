from .types import *
from ..graphic import *
from ..base import *
from .to_runtime.train_model import *
from .to_runtime.align_x import *


def to_models(atn: TrainingAttention, output_train_shapes: list[list[Shape]],
              x_train: list[Grid], params: GlobalParams)->list[AttentionModel]:
    index_blob = gen_all_possible_index(output_train_shapes, atn.x_cluster_info)
    if index_blob is None:
        return []

    possible_sample_index, possible_x_index = index_blob
    df = create_df(x_train, output_train_shapes,
                   possible_sample_index, possible_x_index)
    label = create_label(atn.sample_index, atn.x_index,
                         possible_sample_index, possible_x_index)
    models = train_model(df, label, params)
    return [AttentionModel(model, atn.x_cluster_info, atn.extra_shapes,
                           atn.syntactic_model) for model in models]


def to_runtimes(
        model: AttentionModel, output_test_shapes: list[list[Shape]],
        x_test: list[Grid])->Optional[InferenceAttention]:
    index_blob = gen_all_possible_index(output_test_shapes, model.x_cluster_info)
    if index_blob is None:
        return None

    sample_index, x_index = index_blob
    df = create_df(x_test, output_test_shapes, sample_index, x_index)
    correct_index = model.model.predict_bool(df)

    result_sample_index, result_x_index = [], []
    for correct, sample_index, index in zip(correct_index, sample_index, x_index):
        if correct:
            result_sample_index.append(sample_index)
            result_x_index.append(index)
            # TODO handle syntactic
    return InferenceAttention(result_sample_index, result_x_index, model.extra_shapes)
