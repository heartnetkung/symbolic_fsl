from .abstract_modeling import *
from typing import Callable, Any
import logging
import re
from ..graphic import Deduplicator
from .arc_state import ArcTrainingState, ArcInferenceState, ArcState

logger = logging.getLogger(__name__)
COST_PATTERN = re.compile(r'\*|\+|\- |<|>|==|<=|>=')


def default_cost(obj: Any)->int:
    score = 0
    for k, v in obj.__dict__.items():
        # a hack to count all complex symbols in machine learning models
        score += len(COST_PATTERN.split(repr(v)))-1
    return score


class ModelFreeTask(Task[TS], ModeledTask[IS], InferenceTask):
    '''
    A task without modeling capability which means
    it stays the same throughout lifecycle.
    '''

    def to_models(self, before: TS, after: TS)->list[ModeledTask]:
        return [self]

    def to_runtimes(self, before: IS)->Optional[InferenceTask]:
        return self

    def get_cost(self)->int:
        return 0

    def to_inference(self)->InferenceTask:
        return self


class ModelFreeArcAction(
        Action[ArcTrainingState, Task], InferenceAction[ArcInferenceState, IT]):
    '''
    An action without model which means it stays the same throughout lifecycle.
    Use isinstance(state, ArcTrainingState) to access training-time features.
    '''

    def to_runtimes(self, before: ArcTrainingState, after: ArcTrainingState,
                    task: Task)->list[InferenceAction]:
        return [self]

    def perform_train(self, state: ArcTrainingState,
                      task: Task)->Optional[ArcTrainingState]:
        return self.perform(state, task.to_inference())  # type:ignore

    def perform_infer(self, state: ArcInferenceState,
                      task: IT)->Optional[ArcInferenceState]:
        return self.perform(state, task)  # type:ignore

    def get_cost(self)->int:
        return 0

    @abstractmethod
    def perform(self, state: ArcState, task: IT)->Optional[ArcState]:
        pass


class ModelBasedArcAction(
        Action[ArcTrainingState, T], InferenceAction[ArcInferenceState, IT]):
    '''
    An action with model of how it will be applied during runtime.
    '''

    def perform_train(self, state: ArcTrainingState,
                      task: T)->Optional[ArcTrainingState]:
        return self.perform(state, task.to_inference())  # type:ignore

    def perform_infer(self, state: ArcInferenceState,
                      task: IT)->Optional[ArcInferenceState]:
        return self.perform(state, task)  # type:ignore

    def get_cost(self)->int:
        return default_cost(self)

    def to_runtimes(self, before: ArcTrainingState, after: ArcTrainingState,
                    task: T)->list[InferenceAction]:
        return self.train_models(before, task)  # type:ignore

    @abstractmethod
    def perform(self, state: ArcState, task: IT)->Optional[ArcState]:
        pass

    @abstractmethod
    def train_models(self, state: ArcTrainingState, task: T)->list[InferenceAction]:
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


class TrainingOnlyAction(Action[TS, T]):
    '''An action that only performs during training.'''

    def to_runtimes(self, before: TS, after: TS, task: T)->list[InferenceAction]:
        return [DoNothing()]


class DoNothing(InferenceAction[IS, IT]):
    def perform_infer(self, state: IS, task: IT)->Optional[IS]:
        return state

    def get_cost(self)->int:
        return 0
