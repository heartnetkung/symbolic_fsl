from ...base import *
from ...graphic import *
from ...attention import *
from ..attention_task import TrainingAttentionTask
from .util import *


class AttentionManager(Manager[ArcTrainingState]):
    def __init__(self, common_finder: CommonYFinder, params: GlobalParams)->None:
        self.common_finder = common_finder
        self.global_maker = GlobalAttentionMaker()
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

        common_y_shapes = self.common_finder.find_common_y(state.y_shapes)
        g_atn = self.global_maker.make_global_attentions(state)
        for attention in attentions:
            new_state = state.update(attention_cache=attention)
            new_attention = TrainingAttentionTask(
                attention, g_atn, common_y_shapes, self.params)
            results.append((new_attention, new_state))
        return results
