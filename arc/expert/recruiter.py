from ..base import BasicRecruiter, GlobalParams
from ..manager.task import *

from .parser import *
from .post_loop import *
# no import star except manager.task, reparser.experts, and subfolders

class ArcRecruiter(BasicRecruiter):
    def __init__(self, params: GlobalParams)->None:
        expert_directory = {
            DrawCanvasTask: [DrawCanvasExpert(params)],
        }
        super().__init__(expert_directory)  # type:ignore