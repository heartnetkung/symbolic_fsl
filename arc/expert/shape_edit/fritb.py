from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from copy import deepcopy
from ..util import *
from enum import Enum


class FRITBCondition(Enum):
    min_size = 0
    largest = 1


MAX_ITR = 60


class FillRectangleInTheBlank(ModelFreeArcAction[AttentionTask]):
    def __init__(self, condition: FRITBCondition, feat_index: int,
                 color: int, min_width_height: int, params: GlobalParams)->None:
        self.condition = condition
        self.feat_index = feat_index
        self.color = color
        self.min_width_height = min_width_height
        self.params = params

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes is not None

        atn = task.atn
        result = deepcopy(state.out_shapes)

        for id1, shape_ids in zip(atn.sample_index, atn.x_index):
            id2 = shape_ids[self.feat_index]
            shape = result[id1][id2]
            new_shape = self._perform(shape, self.color)
            if new_shape is None:
                return None

            result[id1][id2] = new_shape
        return state.update(out_shapes=result)

    def _perform(self, shape: Shape, color: int)->Optional[Shape]:
        if not shape._grid.has_color(NULL_COLOR):
            return None

        min_wh = self.min_width_height
        container = NonOverlapingContainer(shape.width, shape.height)

        for x in range(shape.width - self.min_width_height+1):
            for y in range(shape.height - self.min_width_height+1):
                new_rect = _find_largest_rectangle(shape._grid, x, y, color)
                if new_rect is None:
                    continue
                if (new_rect.width < min_wh) or (new_rect.height < min_wh):
                    continue

                success = container.add(new_rect)
                if not success:
                    overlaps = container.query_overlap(new_rect)
                    all_areas = [shape.width*shape.height for shape in overlaps]
                    new_rect_area = new_rect.width*new_rect.height
                    if new_rect_area > max(all_areas, default=0):
                        for overlap in overlaps:
                            container.remove(overlap)
                        container.add(new_rect)

        if self.condition == FRITBCondition.largest:
            all_areas = [shape.width*shape.height for shape in container.items()]
            largest_area = max(all_areas, default=0)
            candidates = [shape for shape in container.items()
                          if shape.width*shape.height == largest_area]
        elif self.condition == FRITBCondition.min_size:
            candidates = list(container.items())
        else:
            raise Exception('unsupported type')

        result_grid = Grid(deepcopy(shape._grid.data))
        for candidate in candidates:
            candidate.draw(result_grid)
        return Unknown(shape.x, shape.y, result_grid)


def _find_largest_rectangle(grid: Grid, x: int, y: int,
                            color: int)->Optional[FilledRectangle]:
    '''A dynamic programming approach to find_largest_rectangle.'''
    if grid.data[y][x] != NULL_COLOR:
        return None

    width, height = 1, 1
    previous_expand_right, previous_expand_bottom = True, True
    for _ in range(MAX_ITR):
        expand_right = _can_expand(
            grid, x, y, width, height, previous_expand_right, True)
        expand_bottom = _can_expand(
            grid, x, y, width, height, previous_expand_bottom, False)

        if expand_right and expand_bottom:
            if (width+1)*height >= (height+1)*width:
                is_right_selected = True
            else:
                is_right_selected = False
        elif expand_right:
            is_right_selected = True
            previous_expand_bottom = False
        elif expand_bottom:
            is_right_selected = False
            previous_expand_right = False
        else:
            return FilledRectangle(x, y, width, height, color)

        if is_right_selected:
            width += 1
        else:
            height += 1
    return None


def _can_expand(grid: Grid, x: int, y: int, width: int, height: int,
                previous: bool, is_right: bool)->bool:
    if previous == False:
        return False
    if is_right and (x+width+1 > grid.width):
        return False
    if (not is_right) and (y+height+1 > grid.height):
        return False

    if is_right:
        to_check = {grid.data[i][x+width] for i in range(y, y+height)}
    else:
        to_check = {grid.data[y+height][j] for j in range(x, x+width)}
    return to_check == {NULL_COLOR}
