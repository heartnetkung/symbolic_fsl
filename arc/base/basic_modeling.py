from .abstract_modeling import *
from typing import Callable, Any
import logging
import re
from ..graphic import Deduplicator
from .arc_state import ArcTrainingState, ArcState

logger = logging.getLogger(__name__)
COST_PATTERN = re.compile(r'\*|\+|\- |<|>|==')


def default_cost(obj: Any)->int:
    score = 0
    for k, v in obj.__dict__.items():
        # a hack to count all complex symbols in machine learning models
        score += len(COST_PATTERN.split(repr(v)))-1
    return score


class ModelFreeTask(Task[TS], ModeledTask[IS], InferenceTask):
    def to_models(self, before: TS, after: TS)->list[ModeledTask]:
        return [self]

    def to_runtimes(self, before: IS)->Optional[InferenceTask]:
        return self

    def get_cost(self)->int:
        return 0


class ModelFreeArcAction(Action[TS, T], InferenceAction[IS, IT]):
    def to_runtimes(self, before: TS, after: TS, task: T)->list[InferenceAction]:
        return [self]

    def perform_train(self, state: TS, task: T)->Optional[TS]:
        return self.perform(state, True)  # type:ignore

    def perform_infer(self, state: IS, task: IT)->Optional[IS]:
        return self.perform(state, False)  # type:ignore

    def get_cost(self)->int:
        return 0

    @abstractmethod
    def perform(self, state: ArcState, is_training: bool)->Optional[ArcState]:
        pass


class BasicRecruiter(Recruiter):
    def __init__(self, mapping: dict[type[Task], list[Expert]])->None:
        self.mapping = mapping

    def recruit(self, task: Task)->list[Expert]:
        try:
            return self.mapping[task.__class__]
        except KeyError:
            print(f'implementation missing: {task.__class__}')
            raise


class ArcSuccessCriteria(SuccessCriteria[ArcTrainingState]):
    def is_success(self, state: ArcTrainingState)->bool:
        return state.out == state.y
