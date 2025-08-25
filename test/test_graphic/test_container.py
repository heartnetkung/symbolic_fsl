from ...arc.graphic import *


def test_add():
    container = NonOverlapingContainer(10, 10)
    success = container.add(FilledRectangle(-1, -1, 5, 5, 1))
    assert not success
    assert len(container) == 0

    success = container.add(FilledRectangle(0, 0, 5, 5, 1))
    assert success
    assert len(container) == 1

    success = container.add(FilledRectangle(1, 1, 6, 6, 2))
    assert not success
    assert len(container) == 1


def test_query():
    container = NonOverlapingContainer(10, 10)
    container.add(FilledRectangle(0, 0, 5, 5, 1))
    container.add(FilledRectangle(5, 5, 5, 5, 2))
    assert len(container) == 2

    result = container.query_overlap(FilledRectangle(1, 1, 1, 1, 3))
    assert result == [FilledRectangle(0, 0, 5, 5, 1)]

    result = container.query_overlap(FilledRectangle(3, 3, 4, 4, 4))
    assert result == [FilledRectangle(0, 0, 5, 5, 1), FilledRectangle(5, 5, 5, 5, 2)]

    container.remove(FilledRectangle(5, 5, 5, 5, 2))
    result = container.query_overlap(FilledRectangle(3, 3, 4, 4, 4))
    assert result == [FilledRectangle(0, 0, 5, 5, 1)]


def test_remove():
    container = NonOverlapingContainer(10, 10)
    container.add(FilledRectangle(0, 0, 5, 5, 1))
    container.add(FilledRectangle(5, 5, 5, 5, 2))
    assert len(container) == 2

    success = container.remove(FilledRectangle(3, 3, 4, 4, 4))
    assert not success
    assert len(container) == 2

    success = container.remove(FilledRectangle(5, 5, 5, 5, 2))
    assert success
    assert len(container) == 1

    success = container.add(FilledRectangle(5, 5, 5, 5, 2))
    assert success
    assert len(container) == 2
