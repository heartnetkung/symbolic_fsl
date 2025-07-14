from ..base import Task, ArcTrainingState, UniversalTask
from dataclasses import dataclass
from typing import Callable
# no import star

# =========================
# parse reparse
# =========================


@dataclass(frozen=True)
class ParseGridTask(UniversalTask):
    pass

# =========================
# finish up
# =========================


@dataclass(frozen=True)
class DrawCanvasTask(UniversalTask):
    pass