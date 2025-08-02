from __future__ import annotations
from .abstract_modeling import TrainingState, InferenceState
from dataclasses import dataclass, field, replace, fields, asdict
from typing import Optional, Any, Union
from ..graphic import Shape, Grid
from ..constant import default_hash
from ..attention import Attention
from functools import cached_property


@dataclass(frozen=True)
class ArcTrainingState(TrainingState[Grid, Grid]):

    # inherited
    x: list[Grid] = field(repr=False, compare=False)
    y: list[Grid] = field(repr=False, compare=False)
    out: Optional[list[Grid]] = None

    # info
    x_bg: Optional[list[int]] = None
    y_bg: Optional[list[int]] = None
    has_layer: bool = False
    reparse_count: int = field(repr=False, compare=False, default=0)

    # shapes
    x_shapes: Optional[list[list[Shape]]] = field(repr=False, default=None)
    y_shapes: Optional[list[list[Shape]]] = None
    out_shapes: Optional[list[list[Shape]]] = None

    # checklist for parsing-related logic, exclusively used by ArcManager
    edge_reparse: bool = False
    merge_nearby_reparse: bool = False
    stack_reparse: bool = False
    split_reparse: bool = False
    run_physics: bool = False
    partitionless_logic: bool = False

    # attention cache, exclusively used by ArcManager
    attention_cache: Optional[Attention] = field(
        repr=False, default=None, compare=False)

    def update(self, **kwargs)->ArcTrainingState:
        return replace(self, **kwargs)

    def check_all(self)->ArcTrainingState:
        return replace(self, edge_reparse=True, merge_nearby_reparse=True,
                       stack_reparse=True, split_reparse=True, run_physics=True,
                       partitionless_logic=True)

    @cached_property
    def _hash(self)->int:
        return default_hash(self)

    def __hash__(self)->int:
        return self._hash


@dataclass(frozen=True)
class ArcInferenceState(InferenceState[Grid, Grid]):
    # inherited
    x: list[Grid] = field(repr=False, compare=False)
    out: Optional[list[Grid]] = None

    # info
    x_bg: Optional[list[int]] = None
    y_bg: Optional[list[int]] = None
    has_layer: bool = False

    # shapes
    x_shapes: Optional[list[list[Shape]]] = field(repr=False, default=None)
    out_shapes: Optional[list[list[Shape]]] = None

    def update(self, **kwargs)->ArcInferenceState:
        return replace(self, **kwargs)

    @cached_property
    def _hash(self)->int:
        return default_hash(self)

    def __hash__(self)->int:
        return self._hash


ArcState = Union[ArcTrainingState, ArcInferenceState]
