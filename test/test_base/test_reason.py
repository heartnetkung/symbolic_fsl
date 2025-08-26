from ...arc.base import *
from .mock_impl import *


def test_basic():
    X_train, X_test, y_train, y_test = [1], [2], [9], [17]
    init_state = MockTrainingState(X_train, y_train, X_train)
    init_state2 = MockInferenceState(X_test, X_test)
    manager = MockManager()
    criteria = ArcSuccessCriteria()
    hr = BasicRecruiter({MockTask: [MockExpert()]})
    result = plan(init_state, manager, hr, criteria, 5, 100, 100)
    result2 = adjust(result.plan, init_state2, 10, 10, 999)

    assert len(result2.traces) == 4
    assert result2.traces[0].cost == 4
    assert result2.traces[1].cost == 4
    assert result2.traces[2].cost == 5
    assert result2.traces[3].cost == 8

    assert result2.traces[0].prediction.out == [13]
    assert result2.traces[1].prediction.out == [17]
    assert result2.traces[2].prediction.out == [11]
    assert result2.traces[3].prediction.out == [10]


def test_max_result():
    X_train, X_test, y_train, y_test = [1], [2], [9], [17]
    init_state = MockTrainingState(X_train, y_train, X_train)
    init_state2 = MockInferenceState(X_test, X_test)
    manager = MockManager()
    criteria = ArcSuccessCriteria()
    hr = BasicRecruiter({MockTask: [MockExpert()]})
    result = plan(init_state, manager, hr, criteria, 5, 100, 100)
    result2 = adjust(result.plan, init_state2, 2, 10, 999)

    assert len(result2.traces) == 2
    assert result2.traces[0].cost == 4
    assert result2.traces[1].cost == 4

    assert result2.traces[0].prediction.out == [13]
    assert result2.traces[1].prediction.out == [17]


def test_max_path():
    X_train, X_test, y_train, y_test = [1], [2], [9], [17]
    init_state = MockTrainingState(X_train, y_train, X_train)
    init_state2 = MockInferenceState(X_test, X_test)
    manager = MockManager()
    criteria = ArcSuccessCriteria()
    hr = BasicRecruiter({MockTask: [MockExpert()]})
    result = plan(init_state, manager, hr, criteria, 5, 100, 100)
    result2 = adjust(result.plan, init_state2, 10, 1, 999)

    assert len(result2.traces) == 3
    assert result2.traces[0].cost == 4
    assert result2.traces[1].cost == 4
    assert result2.traces[2].cost == 5

    assert result2.traces[0].prediction.out == [13]
    assert result2.traces[1].prediction.out == [17]
    assert result2.traces[2].prediction.out == [11]
