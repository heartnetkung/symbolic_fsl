from __future__ import annotations
from typing import TypeVar, Generic, Optional, Any
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from dataclasses import dataclass
from functools import cached_property
from ..constant import GlobalParams


IN = TypeVar('IN')
OUT = TypeVar('OUT')


def default_repr(obj: Any)->str:
    vars_ = [f'{k}={v}' for k, v in obj.__dict__.items() if k[0] != '_']
    return '{}({})'.format(obj.__class__.__name__, ','.join(vars_))


class State(Generic[IN, OUT], ABC):
    '''Immutably encapsulate all state information about the solving problem.'''
    X_train: list[IN]
    X_test: list[IN]
    y_train: list[OUT]
    y_test: Optional[list[OUT]]
    out_train: Optional[list[OUT]] = None
    out_test: Optional[list[OUT]] = None

    @abstractmethod
    def __hash__(self)->int:
        '''Hashability is a requirement.'''
        pass


class RuntimeTask(ABC):
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


class ModeledTask(ABC):
    ''''''
    @abstractmethod
    def to_runtimes(self, test_before: S)->Optional[RuntimeTask]:
        '''Predict new inputs.'''
        pass


class Task(ABC):
    '''
    Immutably encapsulate training information required to solve the problem,
    including the subset of inputs/outputs in the state to work on.
    '''
    @abstractmethod
    def to_models(self, train_before: S, train_after: S)->list[ModeledTask]:
        '''For any parameterized task, train the models to predict new inputs.'''
        pass

    def __repr__(self)->str:
        return default_repr(self)


S = TypeVar('S', bound=State)
T = TypeVar('T', bound=Task)
RT = TypeVar('RT', bound=RuntimeTask)


class Action(Generic[S, T], ABC):
    '''Replayable actions for transforming the training part of the state.'''
    @abstractmethod
    def perform(self, state: S, task: T)->Optional[S]:
        pass

    @abstractmethod
    def to_runtimes(self, before: S, after: S, task: T)->list[RuntimeAction]:
        '''For any parameterized action, train the models to predict new inputs.'''
        pass

    def __repr__(self)->str:
        return default_repr(self)


class RuntimeAction(Generic[S, RT], ABC):
    '''Replayable actions for transforming both train and test parts of the state.'''
    @abstractmethod
    def perform(self, state: S, task: RT)->Optional[S]:
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


class Manager(Generic[S], ABC):
    '''Control the flow of the execution and decide what to do in each iteration.'''
    @abstractmethod
    def decide(self, state: S)->list[tuple[Task, S]]:
        pass


class Recruiter(ABC):
    '''Lookup the best experts for the given task.'''
    @abstractmethod
    def recruit(self, task: Task)->list[Expert]:
        pass


class Expert(Generic[S, T], ABC):
    '''Experts apply domain knowledge to the given state.'''
    @abstractmethod
    def solve_problem(self, state: S, task: T)->list[Action]:
        pass


class Program(Generic[S], ABC):
    '''A test-friendly construct used to run a series of actions.'''
    @abstractmethod
    def run(self, state: S)->Optional[S]:
        pass


class SuccessCriteria(Generic[S], ABC):
    '''Condition for success'''
    @abstractmethod
    def is_success(self, state: S)->bool:
        pass


@dataclass(frozen=True)
class Trace:
    task_actions: list[tuple[RuntimeTask, RuntimeAction]]
    prediction: State

    @cached_property
    def cost(self)->int:
        return Trace.cal_cost(self.task_actions)

    def __repr__(self)->str:
        result = [f'Trace(cost={self.cost}):']
        for task, action in self.task_actions:
            result.append(f'  {task} {action}')
        result.append(f'  {self.prediction.out_test}')
        return '\n'.join(result)

    @staticmethod
    def cal_cost(task_actions: list[tuple[RuntimeTask, RuntimeAction]])->int:
        return sum([t.get_cost() + a.get_cost()for t, a in task_actions])
