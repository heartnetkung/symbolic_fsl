from ..test_action.util import *


def test_basic():
    params = GlobalParams()
    problem_no = 1
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
    parsed_state = run_actions(init_state, parsing_actions)

    def func(df):
        return np.where(df['is_outside(x,y)'] == 1, -1, 4)
    action = FillInTheBlank(expansion=ExpansionMode.top_left,
                            feat_index=0,
                            width_model=ColumnModel('shape0.width'),
                            height_model=ColumnModel('shape0.height'),
                            pixel_model=FunctionModel(func),
                            params=params)
    program = AttentionExpertProgram(action, params)
    result = program.run(parsed_state)
    assert result.out_shapes == result.y_shapes
