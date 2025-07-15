from ..base import BasicRecruiter, GlobalParams
from ..manager.task import *

from .parser import *
from .edit import *
from .post_loop import *
from .reparse import *
from .etc import *
# no import star except manager.task, reparser.experts, and subfolders


class ArcRecruiter(BasicRecruiter):
    def __init__(self, params: GlobalParams)->None:
        expert_directory = {
            ParseGridTask: [ParseGridExpert()],
            CleanUpTask: [CleanUpExpert(params)],
            DrawCanvasTask: [DrawCanvasExpert(params)],
            MergeNearbyTask: [MergeNearbyExpert()],
            ReparseEdgeTask: [ReparseEdgeExpert()],
            ReparseSplitTask: [ReparseSplitExpert()],
            ReparseStackTask: [ReparseStackExpert()],
            TrainingAttentionTask: [
                MoveExpert(params), IntersectExpert(), ColorizeExpert(params)]
        }
        super().__init__(expert_directory)  # type:ignore
