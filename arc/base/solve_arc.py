from .dataset import *
from .abstract_modeling import *
from .basic_modeling import *
from .arc_state import *
from ..graphic import Grid
from dataclasses import dataclass
from .plan import *
from .reason import *
import time
import logging

MAX_TIME_S = 600
MAX_PLAN_DEPTH = 10
MAX_PLAN_ITR = 5000
MAX_REASON_PATH = 10
N_RESULT = 2
DUMMY_PRED = Grid([[0]])
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArcResult:
    _id: str
    X_test_count: int
    elapsed_time_s: float
    planning_result: PlanningResult
    reasoning_result: ReasoningResult
    correct: Optional[bool] = None  # if y_test is in dataset check if it's correct

    def __repr__(self)->str:
        time, correct = self.elapsed_time_s, self.correct
        result = [f'\nArcResult(elapsed_time_s={time:.1f}, correct={correct})']

        for i, grids in enumerate(self.predictions):
            result.append(f'=========== prediction #{i+1} ===========')
            for grid in grids:
                result.append('\n'.join([repr(row) for row in grid.data]))
        return '\n'.join(result)

    @property
    def predictions(self)->list[list[Grid]]:
        result = []
        for trace in self.reasoning_result.traces:
            pred = trace.prediction.out
            assert pred is not None
            result.append(pred)
        return result


def solve_arc(
        dataset: Dataset, manager: Manager, hr: Recruiter,
        max_plan_depth: int = MAX_PLAN_DEPTH, max_plan_itr: int = MAX_PLAN_ITR,
        max_reason_path: int = MAX_REASON_PATH, n_result: int = N_RESULT,
        max_time_s: int = MAX_TIME_S)->ArcResult:

    start_time, _id, X_test_count = time.time(), dataset._id, len(dataset.X_test)
    criteria = ArcSuccessCriteria()

    # plan
    planning_result = plan(dataset.to_training_state(), manager, hr, criteria,
                           max_plan_depth, max_plan_itr, max_time_s)
    time_left = int(max_time_s - (time.time()-start_time))
    if logger.isEnabledFor(logging.INFO):
        planning_result.plan.trim()
    logger.info('successful plans: %s', planning_result.plan)

    # reason
    reasoning_result = reason(planning_result.plan, dataset.to_inference_state(),
                              n_result, max_reason_path, time_left)
    elapsed_time = time.time()-start_time

    if dataset.y_test is None:
        correct = None
    else:
        correct = False
        for trace in reasoning_result.traces:
            correct |= trace.prediction.out == dataset.y_test
    return ArcResult(_id, X_test_count, elapsed_time, planning_result,
                     reasoning_result, correct)


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
