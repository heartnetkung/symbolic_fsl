from .shape_queries.make_shape_queries import *
from .types import *
from itertools import product


def make_gattention(x_train_shapes: list[list[Shape]],
                    x_train: list[Grid])->TrainingGlobalAttention:
    query_results = make_shape_queries(x_train, x_train_shapes)
    return TrainingGlobalAttention(tuple(query_results))


def to_gmodel(atn: TrainingGlobalAttention, x_train_shapes: list[list[Shape]],
              x_train: list[Grid])->GlobalAttentionModel:
    all_models = tuple([query.models for query in atn.shape_queries])
    return GlobalAttentionModel(all_models)


def to_gruntimes(atn_model: GlobalAttentionModel, x_test_shapes: list[list[Shape]],
                 x_test: list[Grid])->list[InferenceGlobalAttention]:
    if len(atn_model.query_models) == 0:
        return [InferenceGlobalAttention(tuple())]

    df = make_query_df(x_test, x_test_shapes)
    all_shape_queries, len_ = [], len(x_test)
    for models in atn_model.query_models:
        query = {}
        for model in models:
            try:
                selection = model.predict_bool(df)
                if check_query_result(df, len_, selection):
                    query[tuple(selection)] = (model, selection)
            except KeyError:
                pass

        if len(query) == 0:
            all_shape_queries.append([create_null_shape_query(len_)])
        else:
            new_queries = []
            for model, selection in query.values():
                selected_rows = df[selection]
                sample_index = tuple(selected_rows['sample_index'].astype(int))
                shape_index = tuple(selected_rows['shape_index'].astype(int))
                new_queries.append(ShapeQuery(sample_index, shape_index, model))
            all_shape_queries.append(new_queries)

    return [InferenceGlobalAttention(query_comb)
            for query_comb in product(*all_shape_queries)]
