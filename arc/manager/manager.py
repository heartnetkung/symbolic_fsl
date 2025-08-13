from __future__ import annotations
from ..base import *
from ..graphic import *
from .task import *
import logging
from ..attention import *
from .reparse.reparse_creator import *
from .submanager.crop_manager import CropManager
from ..algorithm.find_shapes import *


class ArcManager(Manager[ArcTrainingState]):
    '''
    Manager is responsible for deciding what to do in each planning iteration.
    Do not reuse manager across different problems
    '''

    def __init__(self, params: GlobalParams):
        self.params = params
        self.crop_manager = CropManager()

    def decide(self, state: ArcTrainingState)->list[
            tuple[Task[ArcTrainingState], ArcTrainingState]]:

        # parse, reparse, and operations that can run once
        if state.x_shapes is None:
            return [(ParseGridTask(), state)]
        if state.run_physics is False:
            return [(PhysicsTask(), state.update(run_physics=True))]
        if state.partitionless_logic is False:
            return [(PartitionlessLogicTask(), state.update(partitionless_logic=True))]
        if state.free_draw is False:
            return [(FreeDrawTask(), state.update(free_draw=True))]
        if state.edge_reparse is False:
            return [(create_reparse_edge(state), state.update(edge_reparse=True))]
        if state.merge_nearby_reparse is False:
            return ([(MergeNearbyTask(), state.update(merge_nearby_reparse=True))])
        if state.stack_reparse is False:
            return [(ReparseStackTask(), state.update(stack_reparse=True))]
        if state.split_reparse is False:
            return [(create_reparse_split(state), state.update(split_reparse=True))]

        # finish up
        if state.out is not None:
            return []
        if _all_shapes_matched(state.out_shapes, state.y_shapes):
            return [(DrawCanvasTask(), state)]
        if _all_shapes_subset(state.out_shapes, state.y_shapes):
            return [(CleanUpTask(), state)]

        attentions = self._make_attention_task(state)
        draw_lines = _to_task_states(state, make_line_tasks(state, self.params))
        crop_tasks = self.crop_manager.decide(state)
        return attentions+draw_lines+crop_tasks

    def _make_attention_task(self, state: ArcTrainingState)->list[
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

        common_y_shapes = find_common_y_shapes(state.y_shapes)
        for attention in attentions:
            new_state = state.update(attention_cache=attention)
            new_attention = TrainingAttentionTask(
                attention, common_y_shapes, self.params)
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


def _to_task_states(
    state: ArcTrainingState, tasks: list[Task[ArcTrainingState]])->list[
        tuple[Task[ArcTrainingState], ArcTrainingState]]:
    return [(task, state) for task in tasks]
