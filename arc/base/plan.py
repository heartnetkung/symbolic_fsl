from .abstract_modeling import *
from typing import Optional
from dataclasses import dataclass
import time
import logging
from .planning_graph import PlanningGraph
from ..graphic import Deduplicator
from collections import deque


logger = logging.getLogger(__name__)
DEBUG_ITR: set[int] = set()


@dataclass(frozen=True)
class PlanningResult:
    plan: PlanningGraph
    iteration_count: int
    message: str


class StateQueue:
    def __init__(self, dedup: Deduplicator)->None:
        self.data = deque()
        self.dedup = dedup

    def queue(self, state: TrainingState)->bool:
        if self.dedup.has_seen_before(state):
            return False
        self.data.append(state)
        return True

    def dequeue(self)->Optional[TrainingState]:
        try:
            return self.data.popleft()
        except IndexError:
            return None

    def __len__(self)->int:
        return len(self.data)


def plan(initial_state: TrainingState, manager: Manager, hr: Recruiter,
         criteria: SuccessCriteria, max_depth: int, max_iteration: int,
         max_time_s: int)->PlanningResult:

    end_time = time.time()+max_time_s
    dedup = Deduplicator()
    current_queue, next_queue = StateQueue(dedup), StateQueue(dedup)
    current_queue.queue(initial_state)
    plan = PlanningGraph(initial_state)
    iteration_no = 0

    for depth in range(max_depth):
        while True:
            iteration_no += 1
            logger.info('\n========= iteration: %d', iteration_no)

            if time.time() > end_time:
                logger.info('time limit')
                return PlanningResult(plan, iteration_no, 'time limit')

            if iteration_no > max_iteration:
                logger.info('iteration limit reached')
                return PlanningResult(plan, iteration_no, 'iteration limit reached')

            state = current_queue.dequeue()
            if state is None:
                break

            if iteration_no in DEBUG_ITR:
                logger.info('state: %s', state)

            try:
                task_states = manager.decide(state)
                if len(task_states) == 0:
                    logger.info('no task')
            except Exception:
                logger.info('manager.decide error', exc_info=True)
                continue

            for task, local_state in task_states:
                logger.info('task: %s', task)
                for expert in hr.recruit(task):
                    new_action_states = _dispatch_expert(task, local_state, expert)
                    for new_action, new_state in new_action_states:
                        soltion_success = criteria.is_success(new_state)
                        next_queue.queue(new_state)
                        add_success = plan.add_state(
                            state, new_state, task, new_action, soltion_success)

                        if add_success:
                            logger.info('new action: %s', new_action)
                        if soltion_success:
                            logger.info('solution success')

        if len(next_queue) == 0:
            logger.info('options exhausted')
            return PlanningResult(plan, iteration_no, 'options exhausted')
        current_queue, next_queue = next_queue, current_queue
        logger.info('next queue size: %d', len(current_queue))

    logger.info('depth limit reached')
    return PlanningResult(plan, iteration_no, 'depth limit reached')


def _dispatch_expert(task: Task, state: TrainingState,
                     expert: Expert)->list[tuple[Action, TrainingState]]:
    try:
        new_actions = expert.solve_problem(state, task)
    except Exception:
        logger.info('solve_problem error', exc_info=True)
        return []

    result = []
    for new_action in new_actions:
        try:
            new_state = new_action.perform_train(state, task)
            if new_state is not None:
                result.append((new_action, new_state))
        except Exception:
            logger.info('action apply error', exc_info=True)
    return result
