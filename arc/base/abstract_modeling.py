from __future__ import annotations
from typing import TypeVar, Generic, Optional, Any, Union
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from dataclasses import dataclass
from functools import cached_property
from ..constant import GlobalParams, default_repr


IN = TypeVar('IN')
OUT = TypeVar('OUT')


class TrainingState(Generic[IN, OUT], ABC):
    '''Immutably encapsulate all training step information about the task.'''
    x: list[IN]
    out: Optional[list[OUT]]
    y: list[OUT]

    @abstractmethod
    def __hash__(self)->int:
        '''Hashability is a requirement.'''
        pass


class InferenceState(Generic[IN, OUT], ABC):
    '''Immutably encapsulate all inference step information about the task.'''
    x: list[IN]
    out: Optional[list[OUT]]

    @abstractmethod
    def __hash__(self)->int:
        '''Hashability is a requirement.'''
        pass


State = Union[TrainingState, InferenceState]
TS = TypeVar('TS', bound=TrainingState)
IS = TypeVar('IS', bound=InferenceState)
S = TypeVar('S', bound=State)


class InferenceTask(ABC):
    '''
    Immutably encapsulate both train and test information required to solve the problem,
    including the subset of inputs/outputs in the state to work on.
    '''

    def get_cost(self)->int:
        '''
        Used for tie-breaking when there are many solutions.
        Larger score means worse solution.
        Score must not be negative.
        '''
        return 0

    def __repr__(self)->str:
        return default_repr(self)


class ModeledTask(Generic[IS], ABC):
    ''''''
    @abstractmethod
    def to_runtimes(self, before: IS)->Optional[InferenceTask]:
        '''Predict new inputs.'''
        pass


class Task(Generic[TS], ABC):
    '''
    Immutably encapsulate training information required to solve the problem,
    including the subset of inputs/outputs in the state to work on.
    '''
    @abstractmethod
    def to_models(self, before: TS, after: TS)->list[ModeledTask]:
        '''For any parameterized task, train the models to predict new inputs.'''
        pass

    def __repr__(self)->str:
        return default_repr(self)


T = TypeVar('T', bound=Task)
IT = TypeVar('IT', bound=InferenceTask)


class Action(Generic[TS, T], ABC):
    '''Replayable actions for transforming the training part of the state.'''
    @abstractmethod
    def perform(self, state: TS, task: T)->Optional[S]:
        pass

    @abstractmethod
    def to_runtimes(self, before: TS, after: TS, task: T)->list[InferenceAction]:
        '''For any parameterized action, train the models to predict new inputs.'''
        pass

    def __repr__(self)->str:
        return default_repr(self)


class InferenceAction(Generic[IS, IT], ABC):
    '''Replayable actions for transforming both train and test parts of the state.'''
    @abstractmethod
    def apply(self, state: IS, task: IT)->Optional[IS]:
        pass

    def get_cost(self)->int:
        '''
        Used for tie-breaking when there are many solutions.
        Larger score means worse solution.
        Score must not be negative.
        '''
        return 0

    def __repr__(self)->str:
        return default_repr(self)


class Manager(Generic[TS], ABC):
    '''Control the flow of the execution and decide what to do in each iteration.'''
    @abstractmethod
    def decide(self, state: TS)->list[tuple[Task[TS], TS]]:
        pass


class Recruiter(ABC):
    '''Lookup the best experts for the given task.'''
    @abstractmethod
    def recruit(self, task: Task)->list[Expert]:
        pass


class Expert(Generic[TS, T], ABC):
    '''Experts apply domain knowledge to the given state.'''
    @abstractmethod
    def solve_problem(self, state: TS, task: T)->list[Action]:
        pass


class Program(Generic[IS], ABC):
    '''A test-friendly construct used to run a series of actions.'''
    @abstractmethod
    def run(self, state: IS)->Optional[IS]:
        pass


class SuccessCriteria(Generic[TS], ABC):
    '''Condition for success'''
    @abstractmethod
    def is_success(self, state: TS)->bool:
        pass


@dataclass(frozen=True)
class Trace:
    task_actions: list[tuple[InferenceTask, InferenceAction]]
    prediction: InferenceState

    @cached_property
    def cost(self)->int:
        return Trace.cal_cost(self.task_actions)

    def __repr__(self)->str:
        result = [f'Trace(cost={self.cost}):']
        for task, action in self.task_actions:
            result.append(f'  {task} {action}')
        result.append(f'  {self.prediction.out}')
        return '\n'.join(result)

    @staticmethod
    def cal_cost(task_actions: list[tuple[InferenceTask, InferenceAction]])->int:
        return sum([t.get_cost() + a.get_cost()for t, a in task_actions])
