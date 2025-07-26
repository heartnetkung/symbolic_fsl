from .visual_model import *
from ...constant import IgnoredException
from ...graphic import *
from .util import *
from ..model.ml_model import MLModel


class BoundScan(VisualRepresentation):
    '''
    Scan the image, similar to convnet, looking for 4 corners of the bound.
    '''

    def __init__(self, inclusive: bool)->None:
        self.inclusive = inclusive

    def encode_feature(self, grids: list[Grid], feature: pd.DataFrame)->pd.DataFrame:
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

    def encode_label(self, grids: list[Grid], label: np.ndarray)->np.ndarray:
        assert len(grids) == label.shape[0]
        assert label.shape[1] == 4

        result = []
        for grid, bound in zip(grids, label):
            x_min, y_min, width, height = bound
            x_max, y_max = x_min+width-1, y_min+height-1

            if self.inclusive:
                correct_coords = {
                    (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)}
            else:
                correct_coords = {
                    (x_min-1, y_min-1), (x_max+1, y_min-1),
                    (x_min-1, y_max+1), (x_max+1, y_max+1)}

            for x_coord in range(grid.width):
                for y_coord in range(grid.height):
                    result.append(1 if (x_coord, y_coord) in correct_coords else 0)
        return np.array(result)

    def decode_label(self, grids: list[Grid], label: np.ndarray)->np.ndarray:
        area_sum = sum([grid.width*grid.height for grid in grids])
        assert area_sum == len(label)
        if label.sum() != (4*len(grids)):
            raise IgnoredException()

        result, index = [], 0
        for grid in grids:
            pixels = []

            for x_coord in range(grid.width):
                for y_coord in range(grid.height):
                    if np.allclose(label[index], 1):
                        pixels.append(FilledRectangle(x_coord, y_coord, 1, 1, 1))
                    index += 1

            if len(pixels) != 4:
                raise IgnoredException()

            x, y = bound_x(pixels), bound_y(pixels)
            w, h = bound_width(pixels), bound_height(pixels)
            if self.inclusive:
                result.append([x, y, w, h])
            else:
                result.append([x+1, y+1, w-2, h-2])
        return np.array(result)


class BoundScanModel(VisualModel):
    def __init__(self, inner_model: MLModel, rep: BoundScan)->None:
        self.inner_model = inner_model
        super().__init__(rep)

    def _to_code(self) -> str:
        return f'BoundScanModel(inner_model={self.inner_model.code})'

    def _raw_predict(self, grids: list[Grid], X: pd.DataFrame)->np.ndarray:
        return self.inner_model.predict(X)
