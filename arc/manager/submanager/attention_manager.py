from ...base import *
from ...graphic import *
from ..attention_task import TrainingAttentionTask
from ...algorithm.find_shapes import *
from ...attention import *


class AttentionManager(Manager[ArcTrainingState]):
    def __init__(self, params: GlobalParams)->None:
        self.common_shapes_cache: dict[str, tuple[Shape, ...]] = {}
        self.params = params

    def decide(self, state: ArcTrainingState)->list[
            tuple[Task[ArcTrainingState], ArcTrainingState]]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None
        cache = state.attention_cache

        results, attentions = [], []
        if cache is not None:
            assert isinstance(cache, TrainingAttention)
            is_solved = is_attention_solved(cache, state.out_shapes, state.y_shapes)
            if is_solved == FuzzyBool.maybe:
                return []
            if is_solved == FuzzyBool.no:
                attentions = remake_attentions(
                    cache, state.out_shapes, state.y_shapes, state.x)

        if attentions == []:
            attentions = make_attentions(state.out_shapes, state.y_shapes, state.x)

        common_y_shapes = self._get_common_shapes(state.y_shapes)
        for attention in attentions:
            new_state = state.update(attention_cache=attention)
            new_attention = TrainingAttentionTask(
                attention, common_y_shapes, self.params)
            results.append((new_attention, new_state))
        return results

    def _get_common_shapes(self, y_shapes: list[list[Shape]])->tuple[Shape, ...]:
        key = repr(y_shapes)
        result = self.common_shapes_cache.get(key, None)
        if result is not None:
            return result

        new_result = find_common_y_shapes(y_shapes)
        self.common_shapes_cache[key] = new_result
        return new_result
