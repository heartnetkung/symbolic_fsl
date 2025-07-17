from ..test_action.util import *


def get_parsed_state():
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
    return run_actions(init_state, parsing_actions)


def test_basic():
    params = GlobalParams()
    parsed_state = get_parsed_state()

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


def test_expert_action():
    params = GlobalParams()
    parsed_state = get_parsed_state()

    attentions = make_attentions(parsed_state.out_shapes,
                                 parsed_state.y_shapes, parsed_state.x)
    task = TrainingAttentionTask(attentions[0],params)
    expert = FillInTheBlankExpert(params)
    actions = expert.solve_problem(parsed_state, task)
    result = actions[0].perform(parsed_state, task)

    for out_shapes, y_shapes in zip(result.out_shapes, result.y_shapes):
        assert out_shapes == y_shapes

    infer_actions = actions[0].train_models(parsed_state, task)
    result2 = infer_actions[0].perform(parsed_state, task)
    for out_shapes, y_shapes in zip(result2.out_shapes, result.y_shapes):
        assert out_shapes == y_shapes
