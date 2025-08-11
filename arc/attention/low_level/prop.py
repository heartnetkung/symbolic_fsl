from ...graphic import *
from typing import Any


def list_properties(a: Shape)->dict[str, Any]:
    '''List commonly useful property in a shape.'''
    result = a.to_input_var()
    del result['shape_type']
    del result['shape_value']
    del result['single_color']
    result['center'] = (a.x+a.width/2, a.y+a.height/2)
    result['colors'] = list_shape_colors(a)
    result |= list_shape_representations(a)
    return result


def list_shape_representations(a: Shape)->dict[str, Any]:
    '''
    List representations of a shape.
    Notably, ones that including color/geometric transformation invariants.
    '''

    result = {}
    grid, colorless_grid = a._grid, a._grid.normalize_color()

    result['shape'] = hash(grid)
    result['transformed_shape'] = sorted([
        hash(transformed_grid) for transformed_grid in geom_transform_all(grid)])

    result['colorless_shape'] = hash(colorless_grid)
    result['colorless_transformed_shape'] = sorted([
        hash(transformed_grid)
        for transformed_grid in geom_transform_all(colorless_grid)])
    return result
