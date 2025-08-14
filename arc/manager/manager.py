from __future__ import annotations
from ..base import *
from ..graphic import *
from .task import *
import logging
from .reparse.reparse_creator import *
from .submanager.crop_manager import CropManager
from .submanager.attention_manager import AttentionManager


class ArcManager(Manager[ArcTrainingState]):
    '''
    Manager is responsible for deciding what to do in each planning iteration.
    Do not reuse manager across different problems
    '''

    def __init__(self, params: GlobalParams):
        self.params = params
        self.crop_manager = CropManager()
        self.atn_manager = AttentionManager(params)

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

        attentions = self.atn_manager.decide(state)
        draw_lines = _to_task_states(state, make_line_tasks(state, self.params))
        crop_tasks = self.crop_manager.decide(state)
        return attentions+draw_lines+crop_tasks


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
