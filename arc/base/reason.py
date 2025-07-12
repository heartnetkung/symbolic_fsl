from .abstract_modeling import *
from typing import Optional
from dataclasses import dataclass
import time
import logging
from .planning_graph import PlanningGraph
from itertools import product


logger = logging.getLogger(__name__)
LARGE_VALUE = 9999


@dataclass(frozen=True)
class ReasoningResult:
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
        if value is None:
            self.min_cost_traces[key] = (trace, 1)
        else:
            existing_trace, count = value
            if trace.cost < existing_trace.cost:
                self.min_cost_traces[key] = (trace, 1)
            elif trace.cost == existing_trace.cost:
                self.min_cost_traces[key] = (existing_trace, count+1)

        all_costs = sorted([trace.cost for trace, _ in self.min_cost_traces.values()])
        if len(all_costs) > self.max_result:
            self.acceptable_cost = all_costs[self.max_result-1]

    def to_list(self)->list[Trace]:
        if len(self.min_cost_traces) == 0:
            return []

        all_traces = sorted(self.min_cost_traces.values(),
                            key=lambda x: (LARGE_VALUE * x[0].cost)-x[1])
        return [trace for trace, _ in all_traces][:self.max_result]


class ModelCache:
    def __init__(self)->None:
        self.cache: dict[tuple[TrainingState, TrainingState], list[
            tuple[ModeledTask, InferenceAction]]] = {}

    def get_models(self, before: TrainingState, after: TrainingState,
                   plan: PlanningGraph)->list[tuple[ModeledTask, InferenceAction]]:
        key = (before, after)
        values = self.cache.get(key, None)
        if values is None:
            new_values: list[tuple[ModeledTask, InferenceAction]] = []
            for task, action in plan.get_edge_data(before, after):
                try:
                    modeled_tasks = task.to_models(before, after)
                    runtime_actions = action.to_runtimes(before, after, task)
                    for m_task, r_action in product(modeled_tasks, runtime_actions):
                        new_values.append((m_task, r_action))
                except Exception:
                    logger.info('modeling error', exc_info=True)
            self.cache[key] = new_values
            return new_values
        return values


def reason(plan: PlanningGraph, init_state: InferenceState, max_result: int,
           max_path: int, max_time_s: int)->ReasoningResult:
    result = ResultCollection(max_result)
    model_cache = ModelCache()
    end_time = time.time()+max_time_s

    for path_no, path in enumerate(plan.shortest_simple_paths()):
        logger.info('\n========= path_no: %d', path_no)
        if path_no > max_path:
            return ReasoningResult(result.to_list(), path_no, 'max_path limit')

        if time.time() > end_time:
            logger.info('time limit')
            return ReasoningResult(result.to_list(), path_no, 'time limit')

        _fill_traces(init_state, path, 0, plan, [], result, model_cache)
    return ReasoningResult(result.to_list(), max_path, 'options exhausted')


def _fill_traces(state: InferenceState, path: list[TrainingState], index: int,
                 plan: PlanningGraph,
                 prefix: list[tuple[InferenceTask, InferenceAction]],
                 result: ResultCollection, cache: ModelCache)->None:
    if index == len(path)-1:
        result.append(Trace(prefix, state))
        return
    if Trace.cal_cost(prefix) > result.acceptable_cost:
        return

    node, next_node, is_end = path[index], path[index+1], index == (len(path)-2)
    task_actions = cache.get_models(node, next_node, plan)
    for modeled_task, runtime_action in task_actions:

        runtime_task = modeled_task.to_runtimes(state)
        if runtime_task is None:
            continue

        new_state = runtime_action.apply(state, runtime_task)
        if new_state is None:
            continue

        new_prefix = prefix + [(runtime_task, runtime_action)]
        _fill_traces(new_state, path, index+1, plan, new_prefix, result, cache)
