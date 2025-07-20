from ..test_action.util import *


def get_parsed_state():
    problem_no = 11
    dataset = read_datasets(DatasetChoice.train_v1)[problem_no]
    init_state = dataset.to_training_state()
    parsing_actions = [
        IndependentParse(x_mode=ParseMode.proximity_diag,
                         y_mode=ParseMode.proximity_diag,
                         x_bg_model=ConstantModel(0),
                         y_bg_model=ConstantModel(0),
                         unknown_background=False),
        ReparseEdge(param=ReparseEdgeParam.skip),
        MergeNearby(param=MergeNearbyParam.skip),
        ReparseStack(param=ReparseStackParam.skip),
        ReparseSplit(param=ReparseSplitParam.skip)
    ]
    return run_actions(init_state, parsing_actions)


def test_basic():
    params = GlobalParams()
    parsed_state = get_parsed_state()

    def func(df):
        result = np.where(df['is_plus_path(x,y)'] == 1, df['plus(x,y)'], -1)
        return np.where(df['is_cross_path(x,y)'] == 1, df['cross(x,y)'], result)
    action = FillInTheBlank(expansion=ExpansionMode.center,
                            feat_index=0,
                            width_model=ConstantModel(5),
                            height_model=ConstantModel(5),
                            pixel_model=FunctionModel(func),
                            params=params)
    program = AttentionExpertProgram(action, params)
    result = program.run(parsed_state)
    assert result.out_shapes == result.y_shapes
