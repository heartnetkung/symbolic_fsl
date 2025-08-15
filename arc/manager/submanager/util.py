from ...algorithm.find_shapes import find_common_y_shapes
from ...graphic import *


class CommonYFinder:
    def __init__(self):
        self.common_shapes_cache: dict[str, tuple[Shape, ...]] = {}

    def find_common_y(self, y_shapes: list[list[Shape]])->tuple[Shape, ...]:
        key = repr(y_shapes)
        result = self.common_shapes_cache.get(key, None)
        if result is not None:
            return result

        new_result = find_common_y_shapes(y_shapes)
        self.common_shapes_cache[key] = new_result
        return new_result
