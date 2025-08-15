from ...base import *
from ...graphic import *
from ..task import ReparseSplitTask
from .util import *


class SplitManager(Manager[ArcTrainingState]):
    def __init__(self, common_finder: CommonYFinder)->None:
        self.common_finder = common_finder

    def decide(self, state: ArcTrainingState)->list[
            tuple[Task[ArcTrainingState], ArcTrainingState]]:
        assert state.y_shapes is not None
        common_y_shapes = self.common_finder.find_common_y(state.y_shapes)
        return [(ReparseSplitTask(common_y_shapes), state)]
