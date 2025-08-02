from ...base import *
from ...graphic import *
from ...ml import *
from copy import deepcopy
from ..util import *
from ...manager.task import PartitionlessLogicTask
from enum import Enum


class PartitionlessLogicParam(Enum):
    skip = 0
    normal = 1


class ApplyPartitionlessLogic(ModelFreeArcAction[PartitionlessLogicTask]):
    def __init__(self, param: PartitionlessLogicParam,
                 params: GlobalParams,
                 color: int = NULL_COLOR,
                 type: LogicType = LogicType.and_,
                 is_horizontal: bool = False)->None:
        self.param = param
        self.params = params
        self.color = color
        self.type = type
        self.is_horizontal = is_horizontal
        super().__init__()

    def perform(self, state: ArcState,
                task: PartitionlessLogicTask)->Optional[ArcState]:
        if self.param == PartitionlessLogicParam.skip:
            return state
        new_out_shapes = self.apply(state)
        if not isinstance(state, ArcTrainingState):
            return state.update(out_shapes=new_out_shapes)

        return state.update(out_shapes=new_out_shapes,
                            reparse_count=self.params.max_reparse)

    def apply(self, state: ArcState)->Optional[list[list[Shape]]]:
        assert state.out_shapes is not None
        if not is_single_fullsize_shape(state.out_shapes, state.x):
            return None

        all_splitted_shapes = split_shapes_equally(
            state.out_shapes, self.is_horizontal)
        if all_splitted_shapes is None:
            return None

        result = []
        for shapes in all_splitted_shapes:
            new_shape = apply_logic(shapes, self.color, self.type)
            result.append([new_shape])
        return result


def is_single_fullsize_shape(all_shapes: list[list[Shape]], grids: list[Grid])->bool:
    for shapes, grid in zip(all_shapes, grids):
        if len(shapes) != 1:
            return False
        shape = shapes[0]
        if (shape.x != 0) or (shape.y != 0):
            return False
        if (shape.width != grid.width) or (shape.height != grid.height):
            return False
    return True


def split_shapes_equally(all_shapes: list[list[Shape]],
                         is_horizontal: bool)->Optional[list[list[Shape]]]:
    result = []
    for shapes in all_shapes:
        if len(shapes) != 1:
            return None

        shape = shapes[0]
        if is_horizontal:
            if shape.height <= shape.width:
                return None
            if shape.height % shape.width != 0:
                return None

            size = round(shape.height/shape.width)
            x_values = [0]*size
            y_values = range(0, shape.height, shape.width)
            w_values = [shape.width]*size
        else:
            if shape.width <= shape.height:
                return None
            if shape.width % shape.height != 0:
                return None

            size = round(shape.width/shape.height)
            x_values = range(0, shape.width, shape.height)
            y_values = [0]*size
            w_values = [shape.height]*size

        grid = shape._grid
        result.append([Unknown(0, 0, grid.crop(x, y, w, w))
                       for x, y, w in zip(x_values, y_values, w_values)])
    return result
