from ..base import Task, ArcTrainingState, ModelFreeTask
from dataclasses import dataclass
from typing import Callable
from .attention_task import TrainingAttentionTask, AttentionTask
from .draw_line import *
from .reparse.reparse_creator import ReparseEdgeTask
from collections.abc import Sequence
from ..graphic import Shape
# no import star unless subfolder


@dataclass(frozen=True)
class ParseGridTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class ReparseStackTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class MergeNearbyTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class ReparseSplitTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class CropTask(ModelFreeTask):
    # if true crop from state.out_shapes else state.x
    crop_from_out_shapes: bool


@dataclass(frozen=True)
class PhysicsTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class FreeDrawTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class CleanUpTask(ModelFreeTask):
    pass


@dataclass(frozen=True)
class DrawCanvasTask(ModelFreeTask):
    def is_finishing_task(self)->bool:
        return True
