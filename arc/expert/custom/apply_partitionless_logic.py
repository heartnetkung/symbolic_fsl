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
                 row_count: int = -1,
                 col_count: int = -1)->None:
        self.param = param
        self.params = params
        self.color = color
        self.type = type
        self.row_count = row_count
        self.col_count = col_count
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
            state.out_shapes, self.row_count, self.col_count)
        if all_splitted_shapes is None:
            return None

        result = []
        for shapes in all_splitted_shapes:
            new_shape = apply_logic(shapes, self.color, self.type)
            if new_shape is None:
                return None

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


def split_shapes_equally(all_shapes: list[list[Shape]], row_count: int,
                         col_count: int)->Optional[list[list[Shape]]]:
    results = []
    for shapes in all_shapes:
        if len(shapes) != 1:
            return None

        shape = shapes[0]
        if shape.height % row_count != 0:
            return None
        if shape.width % col_count != 0:
            return None

        height = round(shape.height/row_count)
        width = round(shape.width/col_count)
        new_result, grid = [], shape._grid

        for i in range(row_count):
            for j in range(col_count):
                x, y = width*j, height*i
                new_result.append(Unknown(x, y, grid.crop(x, y, width, height)))
        results.append(new_result)
    return results
