from __future__ import annotations
from .abstract_modeling import State
from dataclasses import dataclass, field, replace, fields, asdict
from typing import Optional
from ..graphic import Shape, Grid
# from ..attention_to_consistency import AttentionToConsistency


@dataclass(frozen=True)
class ArcState(State[Grid, Grid]):

    # inherited
    X_train: list[Grid] = field(repr=False, compare=False)
    X_test: list[Grid] = field(repr=False, compare=False)
    y_train: list[Grid] = field(repr=False, compare=False)
    y_test: Optional[list[Grid]] = field(repr=False, default=None, compare=False)
    out_train: Optional[list[Grid]] = None
    out_test: Optional[list[Grid]] = None

    # info
    background: Optional[Background] = None
    has_layer: bool = False
    reparse_count: int = field(repr=False, compare=False, default=0)

    # shapes
    X_train_shapes: Optional[list[list[Shape]]] = field(repr=False, default=None)
    y_train_shapes: Optional[list[list[Shape]]] = None
    X_test_shapes: Optional[list[list[Shape]]] = field(repr=False, default=None)

    # output
    output_train_shapes: Optional[list[list[Shape]]] = None
    output_test_shapes: Optional[list[list[Shape]]] = None

    # reparsing checklist, exclusively used by ArcManager
    tile_reparse: bool = False
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
        comparable_fields = [f.name for f in fields(self) if f.compare]
        dict_ = asdict(self)
        values = tuple(repr(dict_[f]) for f in comparable_fields)
        return hash(values)


@dataclass(frozen=True)
class Background:
    X_train: list[int]
    y_train: list[int]
    X_test: list[int]
    y_test: list[int]

    @staticmethod
    def constant(X_train: list[Grid], X_test: list[Grid], value: int)->Background:
        len_train, len_test = len(X_train), len(X_test)
        return Background([value]*len_train, [value]*len_train,
                          [value]*len_test, [value]*len_test)

    @staticmethod
    def x_dynamic(X_train: list[Grid], X_test: list[Grid])->Background:
        train = [grid.get_top_color() for grid in X_train]
        test = [grid.get_top_color() for grid in X_test]
        return Background(train, train, test, test)

    @staticmethod
    def double_constant(X_train: list[Grid], X_test: list[Grid], x_value: int,
                        y_value: int)->Background:
        len_train, len_test = len(X_train), len(X_test)
        return Background([x_value]*len_train, [y_value]*len_train,
                          [x_value]*len_test, [y_value]*len_test)
