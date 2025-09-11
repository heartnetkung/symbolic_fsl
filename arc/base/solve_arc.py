from .dataset import *
from .abstract_modeling import *
from .basic_modeling import *
from .arc_state import *
from ..graphic import Grid
from dataclasses import dataclass
from .plan import *
from .adjust import *
import time
import logging
from ..constant import N_RESULT

MAX_TIME_S = 600
MAX_PLAN_DEPTH = 12
MAX_PLAN_ITR = 5000
MAX_ADJUST_PATH = 500
PLAN_TIME_RATIO = 0.8
DUMMY_PRED = Grid([[0]])
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArcResult:
    _id: str
    X_test_count: int
    elapsed_time_s: float
    planning_result: PlanningResult
    adjusting_result: AdjustingResult
    correct: Optional[bool] = None  # if y_test is in dataset check if it's correct
    correct_trace: Optional[Trace] = None

    def __repr__(self)->str:
        time, correct = self.elapsed_time_s, self.correct
        result = [f'\nArcResult(elapsed_time_s={time:.1f}, correct={correct})']
        for i, grids in enumerate(self.predictions):
            result.append(f'=========== prediction #{i+1} ===========')
            for grid in grids:
                result.append('\n'.join([repr(row) for row in grid.data]))
                result.append('')
        return '\n'.join(result)

    @property
    def predictions(self)->list[list[Grid]]:
        result = []
        for trace in self.adjusting_result.traces:
            pred = trace.prediction.out
            assert pred is not None
            result.append(pred)
        return result


def solve_arc(
        dataset: Dataset, manager: Manager, hr: Recruiter, params: GlobalParams,
        max_plan_depth: int = MAX_PLAN_DEPTH, max_plan_itr: int = MAX_PLAN_ITR,
        MAX_ADJUST_PATH: int = MAX_ADJUST_PATH,
        max_time_s: int = MAX_TIME_S)->ArcResult:

    start_time, _id, X_test_count = time.time(), dataset._id, len(dataset.X_test)
    criteria = ArcSuccessCriteria()
    init_state = dataset.to_training_state()
    init_state, max_plan_depth = _config_init_state(init_state, params, max_plan_depth)

    # plan
    planning_time = int(max_time_s*PLAN_TIME_RATIO)
    planning_result = plan(init_state, manager, hr, criteria,
                           max_plan_depth, max_plan_itr, planning_time)
    time_left = int(max_time_s - (time.time()-start_time))
    logger.info('successful plans: %s', planning_result.plan)

    # adjust
    adjusting_result = adjust(planning_result.plan, dataset.to_inference_state(),
                              N_RESULT, MAX_ADJUST_PATH, time_left)
    elapsed_time = time.time()-start_time
    correct, correct_trace = None, None

    if dataset.y_test is not None:
        correct = False
        for trace in adjusting_result.traces:
            if trace.prediction.out == dataset.y_test:
                correct = True
                correct_trace = trace

    return ArcResult(_id, X_test_count, elapsed_time, planning_result,
                     adjusting_result, correct, correct_trace)


def merge_predictions(results: list[ArcResult])->dict[str, list]:
    prediction = {}
    for result in results:
        padding = [[DUMMY_PRED]*result.X_test_count]*N_RESULT
        padded_pred = result.predictions + padding
        prediction[result._id] = [
            {"attempt_1": [padded_pred[0][i].data],
                "attempt_2":[padded_pred[1][i].data]}
            for i in range(result.X_test_count)]
    return prediction


def _config_init_state(state: ArcTrainingState, params: GlobalParams,
                       max_depth: int)->tuple[ArcTrainingState, int]:
    config, depth = {}, max_depth
    if not params.enable_free_draw:
        config['free_draw'] = True
        depth -= 1
    if not params.enable_edge:
        config['edge_reparse'] = True
        depth -= 1
    if not params.enable_merge:
        config['merge_nearby_reparse'] = True
        depth -= 1
    if not params.enable_stack:
        config['stack_reparse'] = True
        depth -= 1
    if not params.enable_split:
        config['split_reparse'] = True
        depth -= 1

    if len(config) == 0:
        return state, max_depth
    return state.update(**config), depth
