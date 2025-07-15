from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from ..util import *
from copy import deepcopy
from enum import Enum
from ...attention import touch_overlap
from typing import Union

MAX_DISTANCE = 30


class MoveType(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
    toward_1d = 4

    def proceed(self, a: Shape, b: Shape)->bool:
        if self == MoveType.up:
            a.y -= 1
        elif self == MoveType.down:
            a.y += 1
        elif self == MoveType.left:
            a.x -= 1
        elif self == MoveType.right:
            a.x += 1
        elif self == MoveType.toward_1d:
            x_intersect = range_intersect(
                range(a.x, a.x+a.width), range(b.x, b.x+b.width))
            y_intersect = range_intersect(
                range(a.y, a.y+a.height), range(b.y, b.y+b.height))

            # no direction
            if len(x_intersect) == 0 and len(y_intersect) == 0:
                return False
            # two directions
            if len(x_intersect) > 0 and len(y_intersect) > 0:
                return False
            if len(x_intersect) > 0:
                if a.y == b.y:
                    return False
                if a.y > b.y:
                    a.y -= 1
                else:
                    a.y += 1
            elif len(y_intersect) > 0:
                if a.x == b.x:
                    return False
                if a.x > b.x:
                    a.x -= 1
                else:
                    a.x += 1
        else:
            raise Exception('unknown type')
        return True


class UntilType(Enum):
    touch = 0
    overlap = 1

    def has_ended(self, a: Shape, b: Shape)->bool:
        if self == UntilType.overlap:
            # quick check
            x_intersect = range_intersect(
                range(a.x, a.x+a.width), range(b.x, b.x+b.width))
            y_intersect = range_intersect(
                range(a.y, a.y+a.height), range(b.y, b.y+b.height))
            if len(x_intersect) == 0 or len(y_intersect) == 0:
                return False

            # real check
            return 'overlap' in touch_overlap(a, b)

        elif self == UntilType.touch:
            # quick check
            x_intersect = range_intersect(
                range(a.x-1, a.x+a.width+1), range(b.x, b.x+b.width))
            y_intersect = range_intersect(
                range(a.y-1, a.y+a.height+1), range(b.y, b.y+b.height))
            if len(x_intersect) == 0 or len(y_intersect) == 0:
                return False

            # real check
            return 'touch' in touch_overlap(a, b)
        else:
            raise Exception('unknown type')


class MoveUntil(ModelFreeArcAction[AttentionTask]):
    def __init__(self, move_type: MoveType, until_type: UntilType,
                 moving_index: int, until_index: int)->None:
        self.move_type = move_type
        self.until_type = until_type
        self.moving_index = moving_index
        self.until_index = until_index
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes is not None

        result_blob = self.move_all(state, task)
        if result_blob is None:
            return result_blob

        new_out_shapes = deepcopy(state.out_shapes)
        for shape, id1, id2 in zip(*result_blob):
            new_out_shapes[id1][id2] = shape
        return state.update(out_shapes=new_out_shapes)

    def move_all(self, state: ArcState,
                 task: Union[AttentionTask, TrainingAttentionTask])->Optional[
            tuple[list[Shape], list[int], list[int]]]:
        assert state.out_shapes is not None

        out_shapes,atn = state.out_shapes,task.atn
        new_shapes, id1s, id2s = [], [], []
        for id1, shape_ids in zip(atn.sample_index, atn.x_index):
            id_a = shape_ids[self.moving_index]
            id_b = shape_ids[self.until_index]
            shape_a = deepcopy(out_shapes[id1][id_a])
            shape_b = out_shapes[id1][id_b]
            new_shape = _move(shape_a, shape_b, self.move_type, self.until_type)
            if new_shape is None:
                return None

            new_shapes.append(new_shape)
            id1s.append(id1)
            id2s.append(id_a)
        return new_shapes, id1s, id2s


def _move(a: Shape, b: Shape, move_type: MoveType,
          until_type: UntilType)->Optional[Shape]:
    for _ in range(MAX_DISTANCE):
        success = move_type.proceed(a, b)
        if not success:
            return None
        if until_type.has_ended(a, b):
            return a
    return None
