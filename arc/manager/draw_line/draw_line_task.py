from __future__ import annotations
from ...attention import *
from ...base import *
import pandas as pd
import numpy as np
from typing import Optional
from ...ml import generate_df
from ...graphic import Grid, Shape
from dataclasses import dataclass
import math
from .types import *
from .reparse_line import *


@dataclass(frozen=True)
class DrawLineTask(InferenceTask):
    atn: Attention
    all_lines: AllLineShapes

    def get_cost(self)->int:
        return default_cost(self)


@dataclass(frozen=True)
class ModeledDrawLineTask(ModeledTask[ArcInferenceState]):
    model: AttentionModel
    all_lines: AllLineShapes

    def to_runtimes(self, before: ArcInferenceState)->Optional[InferenceTask]:
        assert before.out_shapes is not None
        atn = to_runtimes(self.model, before.out_shapes, before.x)
        if atn is None:
            return None
        return DrawLineTask(atn, self.all_lines)


@dataclass(frozen=True)
class TrainingDrawLineTask(Task[ArcTrainingState]):
    atn: TrainingAttention
    all_lines: AllLineShapes
    params: GlobalParams

    def to_models(self, before: ArcTrainingState,
                  after: ArcTrainingState)->list[ModeledTask]:
        assert before.out_shapes is not None
        models = to_models(self.atn, before.out_shapes, before.x, self.params)
        return [ModeledDrawLineTask(model, self.all_lines) for model in models]

    def to_inference(self)->InferenceTask:
        return DrawLineTask(self.atn, self.all_lines)

    def get_attention_aligned_lines(self)->list[Line]:
        result = []
        for id1, y_index in zip(self.atn.sample_index, self.atn.y_index):
            id2 = int(math.floor(y_index/2))
            result.append(self.all_lines.all_lines[id1][id2])
        return result


class AllLineShapes:
    def __init__(self, y_shapes: list[list[Shape]])->None:
        self.all_shapes: list[list[Shape]] = []
        self.all_lines: list[list[Line]] = []

        for shapes in y_shapes:
            sample_lines, sample_shapes = [], []
            for shape in shapes:
                lines = reparse_line(shape)
                if len(lines) > 0:
                    sample_lines += lines
                else:
                    sample_shapes.append(shape)

            self.all_shapes.append(sample_shapes)
            self.all_lines.append(sample_lines)

    def to_y_shapes(self)->list[list[Shape]]:
        result = []
        for shapes, lines in zip(self.all_shapes, self.all_lines):
            result.append(shapes+[line.to_shape() for line in lines])
        return result

    def to_end_points(self)->list[list[Shape]]:
        results = []
        for lines in self.all_lines:
            new_result = []
            for line in lines:
                start, end = line.get_start_end_pixels()
                new_result.append(start)
                new_result.append(end)
            results.append(new_result)
        return results

    def has_turn(self)->bool:
        for lines in self.all_lines:
            for line in lines:
                if line.has_turn():
                    return True
        return False


def make_line_tasks(state: ArcTrainingState,
                    params: GlobalParams)->list[Task[ArcTrainingState]]:
    assert state.out_shapes is not None
    assert state.y_shapes is not None

    # TODO check for repeat between x and y
    all_line_shapes = AllLineShapes(state.y_shapes)
    if not all_line_shapes.has_turn():
        return []

    end_points = all_line_shapes.to_end_points()
    attentions = make_attentions(state.out_shapes, end_points, state.x)
    filtered_attentions = [atn for atn in attentions
                           if _check_attention(atn, all_line_shapes)]
    return [TrainingDrawLineTask(atn, all_line_shapes, params)
            for atn in filtered_attentions]


def _check_attention(atn: TrainingAttention, db: AllLineShapes)->bool:
    '''
    For an attention to be correct, it must atten to all lines at only one edge.
    '''

    all_db_indexes: set[tuple[int, int]] = set()
    for sample, lines in enumerate(db.all_lines):
        all_db_indexes |= {(sample, index) for index in range(len(lines))}

    all_atn_indexes = []
    for sample, y_index in zip(atn.sample_index, atn.y_index):
        all_atn_indexes.append((sample, int(math.floor(y_index/2))))

    all_atn_indexes_set = set(all_atn_indexes)
    no_repeat = len(all_atn_indexes) == len(all_atn_indexes_set)
    match = all_atn_indexes_set == all_db_indexes
    return no_repeat and match
