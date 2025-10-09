from ..base import BasicRecruiter, GlobalParams
from ..manager.task import *

from .create import *
from .parser import *
from .edit import *
from .post_loop import *
from .etc import *
from .shape_edit import *
from .custom import *
from .convolution import *
# no import star except manager.task and subfolders


class ArcRecruiter(BasicRecruiter):
    def __init__(self, params: GlobalParams)->None:
        expert_directory = {
            ParseGridTask: [ParseGridExpert(params)],
            CleanUpTask: [CleanUpExpert(params)],
            DrawCanvasTask: [DrawCanvasExpert(params)],
            MergeNearbyTask: [MergeNearbyExpert(params)],
            ReparseEdgeTask: [ReparseEdgeExpert(params)],
            ReparseSplitTask: [ReparseSplitExpert(params)],
            ReparseStackTask: [ReparseStackExpert(params)],
            TrainingAttentionTask: [
                MoveExpert(params), ApplyLogicExpert(), ColorizeExpert(params),
                GeomTransformExpert(params), MoveUntilExpert(), CreateExpert(params),
                FillInTheBlankExpert(params), SplitShapeExpert(), FITPExpert(params),
                HammingExpert(params), ConvolutionDrawExpert(params),
                PutObjectExpert(params), SelectOneExpert(params)],
            CropTask: [CropExpert(params)],
            PhysicsTask: [RunPhysicsExpert(params)],
            FreeDrawTask: [FreeDrawExpert(params)]

        }
        super().__init__(expert_directory)  # type:ignore
