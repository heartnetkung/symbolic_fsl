from ...arc.graphic import *


def test_add():
    container = NonOverlapingContainer(10, 10)
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


def test_unknown():
    container = NonOverlapingContainer(10, 10)
    shape_grid = Grid([[-1, 1, -1], [1, 1, 1], [-1, 1, -1]])
    container.add(Unknown(0, 0, shape_grid))
    success = container.add(Unknown(2, 1, shape_grid))
    assert success
    assert len(container) == 2


def test_offgrid():
    container = NonOverlapingContainer(10, 10)
    success = container.add(FilledRectangle(-1, -1, 2, 2, 1))
    assert success
    assert len(container) == 1

    result = container.query_overlap(FilledRectangle(0, 0, 2, 2, 2))
    assert result == [FilledRectangle(-1, -1, 2, 2, 1)]

    success = container.remove(FilledRectangle(-1, -1, 2, 2, 1))
    assert success
    assert len(container) == 0

    success = container.add(FilledRectangle(-2, -2, 2, 2, 3))
    assert not success
    assert len(container) == 0
