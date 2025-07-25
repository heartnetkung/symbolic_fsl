from ...graphic import *
from ...constant import NULL_COLOR, MISSING_VALUE
from scipy.stats import mode


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
