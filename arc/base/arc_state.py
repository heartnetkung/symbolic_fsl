from __future__ import annotations
from .abstract_modeling import TrainingState, InferenceState
from dataclasses import dataclass, field, replace, fields, asdict
from typing import Optional, Any, Union
from ..graphic import Shape, Grid
# from ..attention_to_consistency import AttentionToConsistency


def default_hash(obj: Any)->int:
    comparable_fields = [f.name for f in fields(obj) if f.compare]
    dict_ = asdict(obj)
    values = tuple(repr(dict_[f]) for f in comparable_fields)
    return hash(values)


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

    # reparsing checklist, exclusively used by ArcManager
    stack_reparse: bool = False
    split_reparse: bool = False
    edge_reparse: bool = False
    merge_nearby_reparse: bool = False

    # attention cache, exclusively used by ArcManager
    # attention_cache: Optional[AttentionToConsistency] = field(
    #     repr=False, default=None, compare=False)

    def update(self, **kwargs)->ArcState:
        return replace(self, **kwargs)

    def __hash__(self)->int:
        return default_hash(self)


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

    def update(self, **kwargs)->ArcState:
        return replace(self, **kwargs)

    def __hash__(self)->int:
        return default_hash(self)


ArcState = Union[ArcTrainingState, ArcInferenceState]
