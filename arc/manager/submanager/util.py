from ...algorithm.find_shapes import find_common_y_shapes
from ...graphic import *
from ...global_attention import *
from ...base import *


class CommonYFinder:
    def __init__(self)->None:
        self.common_shapes_cache: dict[str, tuple[Shape, ...]] = {}

    def find_common_y(self, y_shapes: list[list[Shape]])->tuple[Shape, ...]:
        key = repr(y_shapes)
        result = self.common_shapes_cache.get(key, None)
        if result is not None:
            return result

        new_result = find_common_y_shapes(y_shapes)
        self.common_shapes_cache[key] = new_result
        return new_result


class GlobalAttentionMaker:
    def __init__(self)->None:
        self.cache: dict[str, TrainingGlobalAttention] = {}

    def make_global_attentions(self, state: ArcTrainingState)->TrainingGlobalAttention:
        assert state.x_shapes is not None
        key = repr(state.x_shapes)
        value = self.cache.get(key, None)
        if value is not None:
            return value

        new_value = make_gattention(state.x_shapes, state.x)
        self.cache[key] = new_value
        return new_value
