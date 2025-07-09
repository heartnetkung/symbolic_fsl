from ...arc.graphic import *


def test_hollow_rectangle():
    rect1 = HollowRectangle(0, 0, 5, 5, 2, 1)
    canvas = make_grid(5, 5)
    rect1.draw(canvas)
    assert canvas.data == [
        [2, 2, 2, 2, 2],
        [2, -1, -1, -1, 2],
        [2, -1, -1, -1, 2],
        [2, -1, -1, -1, 2],
        [2, 2, 2, 2, 2]]
    objs = list_objects(canvas)
    assert len(objs) == 1
    assert objs[0] == rect1

    rect2 = HollowRectangle(0, 0, 5, 5, 2, 2)
    canvas = make_grid(5, 5)
    rect2.draw(canvas)
    assert canvas.data == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, -1, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2]]
    objs = list_objects(canvas)
    assert len(objs) == 1
    assert objs[0] == rect2


def test_diagonal():
    line1 = Diagonal(0, 0, 4, 2, False)
    canvas = make_grid(5, 5)
    line1.draw(canvas)
    assert canvas.data == [
        [-1, -1, -1, 2, -1],
        [-1, -1, 2, -1, -1],
        [-1, 2, -1, -1, -1],
        [2, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]]
    objs = list_objects(canvas, True)
    assert len(objs) == 1
    assert objs[0] == line1

    line2 = Diagonal(0, 3, 2, 2, True)
    canvas = make_grid(5, 5)
    line2.draw(canvas)
    assert canvas.data == [
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [2, -1, -1, -1, -1],
        [-1, 2, -1, -1, -1]]
    objs = list_objects(canvas, True)
    assert len(objs) == 1
    assert objs[0] == line2

    line3 = Diagonal(1, 1, 4, 2, False)
    canvas = make_grid(5, 5)
    line3.draw(canvas)
    assert canvas.data == [
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 2],
        [-1, -1, -1, 2, -1],
        [-1, -1, 2, -1, -1],
        [-1, 2, -1, -1, -1]]
    objs = list_objects(canvas, True)
    assert len(objs) == 1
    assert objs[0] == line3

    line4 = Diagonal(3, 0, 2, 2, True)
    canvas = make_grid(5, 5)
    line4.draw(canvas)
    assert canvas.data == [
        [-1, -1, -1, 2, -1],
        [-1, -1, -1, -1, 2],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]]
    objs = list_objects(canvas, True)
    assert len(objs) == 1
    assert objs[0] == line4
