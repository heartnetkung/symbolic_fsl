from ...graphic import *
from ...constant import NULL_COLOR, MISSING_VALUE
from scipy.stats import mode
import pandas as pd


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


def encode_conv_feature(grids: list[Grid], feature: pd.DataFrame)->pd.DataFrame:
    assert len(grids) == len(feature)
    records = feature.to_dict('records')
    result = {
        'x': [], 'y': [], 'cell(x,y)': [],
        'cell(x-1,y)': [], 'cell(x,y-1)': [], 'cell(x+1,y)': [], 'cell(x,y+1)': [],
        'line_n(x,y)': [], 'line_s(x,y)': [], 'line_e(x,y)': [], 'line_w(x,y)': [],
        'line_n2(x,y)': [], 'line_s2(x,y)': [],
        'line_e2(x,y)': [], 'line_w2(x,y)': [],
    } | {col: [] for col in feature.columns}

    for grid, record in zip(grids, records):
        for x in range(grid.width):
            for y in range(grid.height):
                result['x'].append(x)
                result['y'].append(y)
                result['cell(x,y)'].append(grid.safe_access(x, y))
                result['cell(x-1,y)'].append(grid.safe_access(x-1, y))
                result['cell(x,y-1)'].append(grid.safe_access(x, y-1))
                result['cell(x+1,y)'].append(grid.safe_access(x+1, y))
                result['cell(x,y+1)'].append(grid.safe_access(x, y+1))
                result['line_s(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y),
                    grid.safe_access(x, y+1), grid.safe_access(x, y+2)]))
                result['line_e(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y),
                    grid.safe_access(x+1, y), grid.safe_access(x+2, y)]))
                result['line_n(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y),
                    grid.safe_access(x, y-1), grid.safe_access(x, y-2)]))
                result['line_w(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y),
                    grid.safe_access(x-1, y), grid.safe_access(x-2, y)]))
                result['line_s2(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y+1), grid.safe_access(x, y+2)]))
                result['line_e2(x,y)'].append(merge_pixels([
                    grid.safe_access(x+1, y), grid.safe_access(x+2, y)]))
                result['line_n2(x,y)'].append(merge_pixels([
                    grid.safe_access(x, y-1), grid.safe_access(x, y-2)]))
                result['line_w2(x,y)'].append(merge_pixels([
                    grid.safe_access(x-1, y), grid.safe_access(x-2, y)]))

                for k, v in record.items():
                    result[k].append(v)
    return pd.DataFrame(result)