from __future__ import annotations
from ...arc.base import *
from dataclasses import dataclass, field, replace, fields, asdict


@dataclass(frozen=True)
class MockState(State[int, int]):
    X_train: list[int]
    X_test: list[int]
    y_train: list[int]
    y_test: Optional[list[int]]
    out_train: Optional[list[int]] = None
    out_test: Optional[list[int]] = None

    def update(self, **kwargs)->MockState:
        return replace(self, **kwargs)

    def __hash__(self)->int:
        comparable_fields = [f.name for f in fields(self) if f.compare]
        dict_ = asdict(self)
        values = tuple(repr(dict_[f]) for f in comparable_fields)
        return hash(values)


class MockTask(UniversalTask):
    pass


class MockManager(Manager[MockState]):
    def decide(self, state: MockState)->list[tuple[Task, MockState]]:
        return [(MockTask(), state)]


class MockExpert(Expert[MockState, MockTask]):
    def solve_problem(self, state: MockState, task: MockTask)->list[Action]:
        return [PlusOneAction(), MulTwoAction()]


class PlusOneRAction(RuntimeAction[MockState, MockTask]):
    def perform(self, state: S, task: RT)->Optional[S]:
        assert state.y_test is not None
        assert state.out_train is not None
        assert state.out_test is not None
        out_test2 = [val+1 for val in state.out_test]
        return state.update(out_test=out_test2)

    def get_cost(self)->int:
        return 1


class PlusOneAction(Action[MockState, MockTask]):
    def perform(self, state: MockState, task: MockTask)->Optional[MockState]:
        assert state.y_test is not None
        assert state.out_train is not None
        assert state.out_test is not None
        out_train2 = [val+1 for val in state.out_train]
        return state.update(out_train=out_train2)

    def to_runtimes(self, before: S, after: S, task: T)->list[RuntimeAction]:
        return [PlusOneRAction()]


class MulTwoRAction(RuntimeAction[MockState, MockTask]):
    def perform(self, state: S, task: RT)->Optional[S]:
        assert state.y_test is not None
        assert state.out_train is not None
        assert state.out_test is not None
        out_test2 = [val*2 for val in state.out_test]
        return state.update(out_test=out_test2)

    def get_cost(self)->int:
        return 1


class MulTwoAction(Action[MockState, MockTask]):
    def perform(self, state: MockState, task: MockTask)->Optional[MockState]:
        assert state.y_test is not None
        assert state.out_train is not None
        assert state.out_test is not None
        out_train2 = [val*2 for val in state.out_train]
        return state.update(out_train=out_train2)

    def to_runtimes(self, before: S, after: S, task: T)->list[RuntimeAction]:
        return [MulTwoRAction()]
