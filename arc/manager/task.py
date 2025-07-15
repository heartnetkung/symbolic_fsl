from ..base import Task, ArcTrainingState, ModelFreeTask
from dataclasses import dataclass
from typing import Callable
from .attention_task import TrainingAttentionTask, AttentionTask
from .reparse.reparse_creator import (
    ReparseStackTask, ReparseSplitTask, ReparseEdgeTask, MergeNearbyTask)
# no import star


@dataclass(frozen=True)
class ParseGridTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class CleanUpTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class DrawCanvasTask(ModelFreeTask):
    pass
