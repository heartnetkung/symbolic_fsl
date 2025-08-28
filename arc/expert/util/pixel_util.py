from ...graphic import *
from ...constant import NULL_COLOR, MISSING_VALUE
from scipy.stats import mode
from typing import Optional


def vote_pixels(pixels: list[int])->int:
    result = []
    for pixel in pixels:
        if not valid_color(pixel):
            continue
        result.append(pixel)
    if len(result) == 0:
        return NULL_COLOR
    return mode(result).mode


def merge_pixels(pixels: list[int])->int:
    result = NULL_COLOR
    for pixel in pixels:
        if not valid_color(pixel):
            continue
        if result == NULL_COLOR:  # first encounter
            result = pixel
        elif result != pixel:  # multiple colors are unacceptable
            return MISSING_VALUE
    return result


def diff_pixels(before: Grid, after: Grid)->Optional[list[tuple[int, int]]]:
    if (before.width, before.height) != (after.width, after.height):
        return None

    result = []
    for i in range(before.height):
        for j in range(before.width):
            cell_b, cell_a = before.data[i][j], after.data[i][j]
            if cell_a != cell_b:
                result.append((cell_b, cell_a))
    return result
