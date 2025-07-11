from .abstract_modeling import *
from typing import Callable, Any
import logging
import re
from ..graphic import Deduplicator
from .arc_state import ArcState

logger = logging.getLogger(__name__)
COST_PATTERN = re.compile(r'\*|\+|\- |<|>|==')


def default_cost(obj: Any)->int:
    score = 0
    for k, v in obj.__dict__.items():
        # a hack to count all complex symbols in machine learning models
        score += len(COST_PATTERN.split(repr(v)))-1
    return score


class UniversalTask(Task, ModeledTask, RuntimeTask):
    def to_models(self, train_before: S, train_after: S)->list[ModeledTask]:
        return [self]

    def to_runtimes(self, test_before: S)->Optional[RuntimeTask]:
        return self


class BasicRecruiter(Recruiter):
    def __init__(self, mapping: dict[type[Task], list[Expert]])->None:
        self.mapping = mapping

    def recruit(self, task: Task)->list[Expert]:
        try:
            return self.mapping[task.__class__]
        except KeyError:
            print(f'implementation missing: {task.__class__}')
            raise


class ArcSuccessCriteria(SuccessCriteria[ArcState]):
    def is_success(self, state: ArcState)->bool:
        return state.out_train == state.y_train
