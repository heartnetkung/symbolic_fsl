from ..test_action.util import *
from ...arc.expert.edit.colorize import colorize


def get_parsed_state():
    problem_no = 5
    dataset = read_datasets(DatasetChoice.train_v1)[problem_no]
    init_state = dataset.to_training_state()
    parsing_actions = [
        IndependentParse(x_mode=ParseMode.partition,
                         y_mode=ParseMode.crop,
                         x_bg_model=ConstantModel(0),
                         y_bg_model=ConstantModel(0),
                         unknown_background=False,
                         x_partition_color=5),
        ReparseEdge(param=ReparseEdgeParam.skip),
        MergeNearby(param=MergeNearbyParam.skip),
        ReparseStack(param=ReparseStackParam.skip),
        ReparseSplit(param=ReparseSplitParam.skip)
    ]
    return run_actions(init_state, parsing_actions)


def test_basic():
    params = GlobalParams()
    parsed_state = get_parsed_state()

    action = ApplyLogic(feat_indexes=[0, 1], color=1,type=LogicType.and_)
    program = AttentionExpertProgram(action, params)
    result = program.run(parsed_state)
    assert result.out_shapes is not None

    for out_shapes, y_shapes in zip(result.out_shapes, result.y_shapes):
        assert colorize(out_shapes[1], 2) == y_shapes[0]
