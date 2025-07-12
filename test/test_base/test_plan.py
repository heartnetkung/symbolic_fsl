from ...arc.base import *
from .mock_impl import *


def test_basic():
    X_train, X_test, y_train, y_test = [1], [2], [9], [17]
    init_state = MockTrainingState(X_train, y_train, X_train)
    manager = MockManager()
    criteria = ArcSuccessCriteria()
    hr = BasicRecruiter({MockTask: [MockExpert()]})
    result = plan(init_state, manager, hr, criteria, 5, 100, 100)
    result.plan.trim()
    assert len(result.plan.end_states) == 1
    assert result.plan.graph.number_of_nodes() == 9
    assert result.plan.graph.number_of_edges() == 11
    assert result.message == 'depth limit reached'


def test_itr_limit():
    X_train, X_test, y_train, y_test = [1], [2], [9], [17]
    init_state = MockTrainingState(X_train, y_train, X_train)
    manager = MockManager()
    criteria = ArcSuccessCriteria()
    hr = BasicRecruiter({MockTask: [MockExpert()]})
    result = plan(init_state, manager, hr, criteria, 5, 1, 100)
    result.plan.trim()
    assert len(result.plan.end_states) == 0
    assert result.plan.graph.number_of_nodes() == 0
    assert result.plan.graph.number_of_edges() == 0
    assert result.message == 'iteration limit reached'


def test_depth_limit():
    X_train, X_test, y_train, y_test = [1], [2], [9], [17]
    init_state = MockTrainingState(X_train, y_train, X_train)
    manager = MockManager()
    criteria = ArcSuccessCriteria()
    hr = BasicRecruiter({MockTask: [MockExpert()]})
    result = plan(init_state, manager, hr, criteria, 1, 100, 100)
    result.plan.trim()
    assert len(result.plan.end_states) == 0
    assert result.plan.graph.number_of_nodes() == 0
    assert result.plan.graph.number_of_edges() == 0
    assert result.message == 'depth limit reached'
