from .shape_queries.make_shape_queries import *
from .types import *


def make_attentions(x_train_shapes: list[list[Shape]],
                    x_train: list[Grid])->list[TrainingGlobalAttention]:
    query_results = make_shape_queries(x_train, x_train_shapes)
    return [TrainingGlobalAttention(q.sample_index, q.shape_index, q.models)
            for q in query_results]


def to_models(atn: TrainingGlobalAttention, x_train_shapes: list[list[Shape]],
              x_train: list[Grid])->list[GlobalAttentionModel]:
    return [GlobalAttentionModel(model) for model in atn.query_models]


def to_runtimes(model: GlobalAttentionModel, x_test_shapes: list[list[Shape]],
                x_test: list[Grid])->Optional[InferenceGlobalAttention]:
    df = make_query_df(x_test, x_test_shapes)
    selection = model.query_model.predict_bool(df)
    if not check_query_result(df, len(x_test), selection):
        return None

    selected_rows = df[selection]
    sample_index = selected_rows['sample_index'].astype(int)
    shape_index = selected_rows['shape_index'].astype(int)
    assert isinstance(sample_index, pd.Series)
    assert isinstance(shape_index, pd.Series)

    return InferenceGlobalAttention(
        sample_index.to_list(), shape_index.to_list(), model.query_model)
