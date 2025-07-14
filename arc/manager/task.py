from ..base import Task, ArcTrainingState, ModelFreeTask
from dataclasses import dataclass
from typing import Callable
# no import star

# =========================
# parse reparse
# =========================


@dataclass(frozen=True)
class ParseGridTask(ModelFreeTask):
    pass

# =========================
# finish up
# =========================


@dataclass(frozen=True)
class DrawCanvasTask(ModelFreeTask):
    pass