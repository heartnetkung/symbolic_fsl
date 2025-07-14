from __future__ import annotations
from ...arc.base import *
from dataclasses import dataclass, field, replace, fields, asdict


@dataclass(frozen=True)
class MockTrainingState(TrainingState[int, int]):
    x: list[int]
    y: list[int]
    out: Optional[list[int]] = None

    def update(self, **kwargs)->MockTrainingState:
        return replace(self, **kwargs)

    def __hash__(self)->int:
        comparable_fields = [f.name for f in fields(self) if f.compare]
        dict_ = asdict(self)
        values = tuple(repr(dict_[f]) for f in comparable_fields)
        return hash(values)


@dataclass(frozen=True)
class MockInferenceState(TrainingState[int, int]):
    x: list[int]
    out: Optional[list[int]] = None

    def update(self, **kwargs)->MockTrainingState:
        return replace(self, **kwargs)

    def __hash__(self)->int:
        comparable_fields = [f.name for f in fields(self) if f.compare]
        dict_ = asdict(self)
        values = tuple(repr(dict_[f]) for f in comparable_fields)
        return hash(values)


class MockTask(ModelFreeTask):
    pass


class MockManager(Manager[MockTrainingState]):
    def decide(self, state: MockTrainingState)->list[tuple[Task, MockTrainingState]]:
        return [(MockTask(), state)]


class MockExpert(Expert[MockTrainingState, MockTrainingState]):
    def solve_problem(self, state: MockTrainingState, task: MockTask)->list[Action]:
        return [PlusOneAction(), MulTwoAction()]


class PlusOneIAction(InferenceAction[MockInferenceState, MockTask]):
    def perform_infer(self, state: MockInferenceState,
                task: MockTask)->Optional[MockInferenceState]:
        return state.update(out=[val+1 for val in state.out])

    def get_cost(self)->int:
        return 1


class PlusOneAction(Action[MockTrainingState, MockTask]):
    def perform_train(self, state: MockTrainingState,
                task: MockTask)->Optional[MockTrainingState]:
        return state.update(out=[val+1 for val in state.out])

    def to_runtimes(self, before: MockTrainingState, after: MockTrainingState,
                    task: MockTask)->list[RuntimeAction]:
        return [PlusOneIAction()]


class MulTwoIAction(InferenceAction[MockInferenceState, MockTask]):
    def perform_infer(self, state: MockInferenceState,
                task: MockTask)->Optional[MockInferenceState]:
        return state.update(out=[val*2 for val in state.out])

    def get_cost(self)->int:
        return 1


class MulTwoAction(Action[MockTrainingState, MockTask]):
    def perform_train(self, state: MockTrainingState,
                task: MockTask)->Optional[MockTrainingState]:
        return state.update(out=[val*2 for val in state.out])

    def to_runtimes(self, before: MockTrainingState, after: MockTrainingState,
                    task: MockTask)->list[RuntimeAction]:
        return [MulTwoIAction()]
