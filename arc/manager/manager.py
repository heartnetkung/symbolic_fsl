from __future__ import annotations
from ..base import *
from ..graphic import *
from .task import *
import logging
from ..attention import *


logger = logging.getLogger(__name__)


class ArcManager(Manager[ArcTrainingState]):
    def __init__(self, params: GlobalParams):
        self.params = params

    def decide(self, state: ArcTrainingState)->list[
            tuple[Task[ArcTrainingState], ArcTrainingState]]:

        # parse and reparse
        if state.x_shapes is None:
            return [(ParseGridTask(), state)]

        # finish up
        if state.out is not None:
            return []
        if _all_shapes_matched(state.out_shapes, state.y_shapes):
            return [(DrawCanvasTask(), state)]

        logger.info('attention_task')
        return self._make_attention_task(state)

    def _make_attention_task(self, state: ArcTrainingState)->list[
            tuple[Task[ArcTrainingState], ArcTrainingState]]:
        assert state.out_shapes is not None
        assert state.y_shapes is not None
        attention = state.attention_cache

        results, attentions = [], []
        if attention is not None:
            assert isinstance(attention, TrainingAttention)
            is_solved = is_attention_solved(
                attention, state.out_shapes, state.y_shapes)
            if is_solved == FuzzyBool.maybe:
                return []
            if is_solved == FuzzyBool.no:
                attentions = remake_attentions(
                    attention, state.out_shapes, state.y_shapes, state.x)

        if attentions == []:
            attentions = make_attentions(state.out_shapes, state.y_shapes, state.x)

        for attention in attentions:
            new_state = state.update(attention_cache=attention)
            new_attention = TrainingAttentionTask(attention, self.params)
            results.append((new_attention, new_state))
        return results


def _all_shapes_matched(a: Optional[list[list[Shape]]],
                        b: Optional[list[list[Shape]]])->bool:
    if a is None or b is None:
        return False

    a_set = [set(shapes) for shapes in a]
    b_set = [set(shapes) for shapes in b]
    return a_set == b_set


def _all_shapes_subset(all_x: Optional[list[list[Shape]]],
                       all_y: Optional[list[list[Shape]]])->bool:
    if all_x is None or all_y is None:
        return False
    for x_shapes, y_shapes in zip(all_x, all_y):
        if not set(y_shapes).issubset(set(x_shapes)):
            return False
    return True
