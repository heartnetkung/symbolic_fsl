from .abstract_modeling import *
from typing import Optional
from dataclasses import dataclass
import time
import logging
from .planning_graph import PlanningGraph
from itertools import product
from ..constant import IgnoredException
import re


logger = logging.getLogger(__name__)
LARGE_VALUE = 9999


@dataclass(frozen=True)
class AdjustingResult:
    traces: list[Trace]
    path_count: int
    message: str


class ResultCollection:
    def __init__(self, max_result: int)->None:
        self.min_cost_traces: dict[str, tuple[Trace, int]] = {}
        self.max_result = max_result
        self.acceptable_cost = LARGE_VALUE

    def append(self, trace: Trace)->None:
        key = repr(trace.prediction.out)
        value = self.min_cost_traces.get(key, None)
        has_new_trace = False

        if value is None:
            self.min_cost_traces[key] = (trace, 1)
            has_new_trace = True
        else:
            existing_trace, count = value
            if trace.cost < existing_trace.cost:
                self.min_cost_traces[key] = (trace, 1)
                has_new_trace = True
            elif trace.cost == existing_trace.cost:
                self.min_cost_traces[key] = (existing_trace, count+1)

        if has_new_trace:
            all_costs = sorted([t.cost for t, _ in self.min_cost_traces.values()])
            if len(all_costs) > self.max_result:
                self.acceptable_cost = all_costs[self.max_result-1]

    def to_list(self)->list[Trace]:
        if len(self.min_cost_traces) == 0:
            return []

        all_traces = sorted(self.min_cost_traces.values(),
                            key=lambda x: (LARGE_VALUE * x[0].cost)-x[1])
        return [trace for trace, _ in all_traces][:self.max_result]

    def __repr__(self)->str:
        result = ['ResultCollection{']
        for key, (trace, count) in self.min_cost_traces.items():
            formatted_grids = re.sub(r'(\[+)', r'\n\1', repr(trace.prediction.out))
            result.append(f'\nprediction: {formatted_grids}')
            result.append(f'count: {count}')
            result.append(f'trace: {trace}')
        return '\n'.join(result+['}'])


class ModelCache:
    def __init__(self, plan: PlanningGraph)->None:
        self.cache = {}
        self.plan = plan

    def get_models(self, before: TrainingState, after: TrainingState)->list[
            tuple[ModeledTask, InferenceAction]]:
        key = (before, after)
        values = self.cache.get(key, None)
        if values is None:
            new_values = self._train_models(before, after)
            self.cache[key] = new_values
            return new_values
        return values

    def _train_models(self, before: TrainingState, after: TrainingState)->list[
            tuple[ModeledTask, InferenceAction]]:
        new_values: list[tuple[ModeledTask, InferenceAction]] = []
        for task, action in self.plan.get_edge_data(before, after):
            try:
                modeled_tasks = task.to_models(before, after)
                runtime_actions = action.to_runtimes(before, after, task)
                for m_task, r_action in product(modeled_tasks, runtime_actions):
                    new_values.append((m_task, r_action))
            except Exception:
                logger.info('modeling error', exc_info=True)
        return new_values


class StateCache:
    def __init__(self, plan: PlanningGraph)->None:
        self.model_cache = ModelCache(plan)
        self.cache = {}

    def get_states(self, train_before: TrainingState, train_after: TrainingState,
                   infer_before: InferenceState)->dict[InferenceState, list[
                       tuple[InferenceTask, InferenceAction]]]:

        key = (train_before, train_after, infer_before)
        values = self.cache.get(key, None)
        if values is None:
            new_values = self._infer_states(train_before, train_after, infer_before)
            self.cache[key] = new_values
            return new_values
        return values

    def _infer_states(self, train_before: TrainingState, train_after: TrainingState,
                      infer_before: InferenceState)->dict[InferenceState, list[
                          tuple[InferenceTask, InferenceAction]]]:

        next_iteration_data = {}
        task_actions = self.model_cache.get_models(train_before, train_after)
        for modeled_task, runtime_action in task_actions:
            try:
                runtime_tasks = modeled_task.to_runtimes(infer_before)

                for runtime_task in runtime_tasks:
                    new_state = runtime_action.perform_infer(infer_before, runtime_task)
                    if new_state is None:
                        continue

                    new_prefix = [(runtime_task, runtime_action)]
                    saved_prefix = next_iteration_data.get(new_state, None)
                    if saved_prefix is None:
                        next_iteration_data[new_state] = new_prefix
                    elif Trace.cal_cost(saved_prefix) > Trace.cal_cost(new_prefix):
                        next_iteration_data[new_state] = new_prefix
            except IgnoredException:
                pass
            except Exception:
                logger.info('runtime action error', exc_info=True)
        return next_iteration_data


def adjust(plan: PlanningGraph, init_state: InferenceState, max_result: int,
           max_path: int, max_time_s: int)->AdjustingResult:
    result = ResultCollection(max_result)
    state_cache = StateCache(plan)
    end_time = time.time()+max_time_s
    last_path_no = 0

    for path_no, path in enumerate(plan.shortest_simple_paths()):
        logger.info('\n========= path_no: %d', path_no)
        last_path_no = path_no
        _print_path(plan, path)

        if path_no > max_path:
            logger.info('max_path limit \n%s', result)
            return AdjustingResult(result.to_list(), path_no+1, 'max_path limit')

        if time.time() > end_time:
            logger.info('time limit \n%s', result)
            return AdjustingResult(result.to_list(), path_no+1, 'time limit')

        _fill_traces(init_state, path, 0, end_time, [], result, state_cache)

    logger.info('options exhausted \n%s', result)
    return AdjustingResult(result.to_list(), last_path_no+1, 'options exhausted')


def _fill_traces(state: InferenceState, path: list[TrainingState], index: int,
                 end_time: float, prefix: list[tuple[InferenceTask, InferenceAction]],
                 result: ResultCollection, cache: StateCache)->None:
    if index == len(path)-1:
        result.append(Trace(prefix, state))
        return
    if time.time() > end_time:
        return
    if Trace.cal_cost(prefix) > result.acceptable_cost:
        return

    node, next_node = path[index], path[index+1]
    next_iteration_data = cache.get_states(node, next_node, state)
    for new_state, new_prefix in next_iteration_data.items():
        _fill_traces(new_state, path, index+1,
                     end_time, prefix+new_prefix, result, cache)


def _print_path(plan: PlanningGraph, path: list[TrainingState])->None:
    if not logger.isEnabledFor(logging.INFO):
        return

    previous_node = path[0]
    for i, node in enumerate(path[1:]):
        logger.info('%d\n%s', i, plan.get_edge_data(previous_node, node))
        previous_node = node
