from ...base import ArcTrainingState, Expert, Action
from .merge_nearby import MergeNearbyParam, MergeNearby
from .reparse_edge import ReparseEdgeParam, ReparseEdge
from .reparse_split import ReparseSplitParam, ReparseSplit
from .reparse_stack import ReparseStackParam, ReparseStack
from ...manager.reparse import *

MAX_REPARSE = 2


class MergeNearbyExpert(Expert[ArcTrainingState, MergeNearbyTask]):
    def solve_problem(self, state: ArcTrainingState,
                      task: MergeNearbyTask)->list[Action]:
        if state.reparse_count >= MAX_REPARSE:
            return [MergeNearby(MergeNearbyParam.skip)]
        return [MergeNearby(param) for param in MergeNearbyParam]


class ReparseEdgeExpert(Expert[ArcTrainingState, ReparseEdgeTask]):
    def solve_problem(self, state: ArcTrainingState,
                      task: ReparseEdgeTask)->list[Action]:
        if state.reparse_count >= MAX_REPARSE:
            return [ReparseEdge(ReparseEdgeParam.skip)]
        return [ReparseEdge(param) for param in ReparseEdgeParam]


class ReparseSplitExpert(Expert[ArcTrainingState, ReparseSplitTask]):
    def solve_problem(self, state: ArcTrainingState,
                      task: ReparseSplitTask)->list[Action]:
        if state.reparse_count >= MAX_REPARSE:
            return [ReparseSplit(ReparseSplitParam.skip)]
        return [ReparseSplit(param) for param in ReparseSplitParam]


class ReparseStackExpert(Expert[ArcTrainingState, ReparseStackTask]):
    def solve_problem(self, state: ArcTrainingState,
                      task: ReparseStackTask)->list[Action]:
        if state.reparse_count >= MAX_REPARSE:
            return [ReparseStack(ReparseStackParam.skip)]
        return [ReparseStack(param) for param in ReparseStackParam]
