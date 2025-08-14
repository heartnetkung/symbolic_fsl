from ...base import *
from ...graphic import *
from ...ml import *
from ...manager.task import *
from copy import deepcopy
from ..util import *
from enum import Enum
from .hamming_generate_df import *


class Hamming(ModelBasedArcAction[TrainingAttentionTask, AttentionTask]):
    def __init__(self, feat_index: int, pixel_model: MLModel,
                 params: GlobalParams)->None:
        self.feat_index = feat_index
        self.pixel_model = pixel_model
        self.params = params
        super().__init__()

    def perform(self, state: ArcState, task: AttentionTask)->Optional[ArcState]:
        assert state.out_shapes is not None

        atn = task.atn
        grids = get_grids(state, atn)
        new_out_shapes = deepcopy(state.out_shapes)

        for grid, id1, shape_ids in zip(grids, atn.sample_index, atn.x_index):
            id2 = shape_ids[self.feat_index]
            shape = new_out_shapes[id1][id2]

            pixel_df = generate_pixel_df([grid], [shape])
            pixel_color = self.pixel_model.predict_int(pixel_df)
            shape_grid = Grid(deepcopy(shape._grid.data))
            for x, y, color in zip(pixel_df['x'], pixel_df['y'], pixel_color):
                shape_grid.safe_assign(x, y, color)

            x, y, shape_grid2 = trim(np.array(shape_grid.data))
            new_out_shapes[id1][id2] = from_grid(shape.x+x, shape.y+y, shape_grid2)

        return state.update(out_shapes=new_out_shapes)

    def train_models(self, state: ArcTrainingState,
                     task: TrainingAttentionTask)->list[InferenceAction]:
        assert isinstance(self.pixel_model, StepMemoryModel)

        x_shapes = get_x_col(state, task.atn, self.feat_index)
        grids = get_grids(state, task.atn)
        df = generate_pixel_df(grids, x_shapes)
        models = make_regressor(
            df, self.pixel_model.result, self.params, 'hamming')
        return [Hamming(self.feat_index, model, self.params) for model in models]


def _gen_dummy_df(grid: Grid)->pd.DataFrame:
    data = {'x': [], 'y': []}
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.data[y][x] != NULL_COLOR:
                data['x'].append(x)
                data['y'].append(y)
    return pd.DataFrame(data)
