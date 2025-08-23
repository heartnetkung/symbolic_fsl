from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from .ml_model import MLModel
import pandas as pd
import numpy as np
from ...constant import GlobalParams
from typing import Callable, Any

UNDEFINED_VALUE = -2
DT_MIN_LEN = 20  # DT is only for powerful prediction or it will overfit
MAX_DEPTH = 3


class DTClassifier(MLModel):
    '''Wrapper class for decision tree classifier'''

    def __init__(self, model: DecisionTreeClassifier, columns: list[str])->None:
        self.model = model
        self.columns = columns

    def predict(self, X: pd.DataFrame)->np.ndarray:
        return self.model.predict(X)

    def _to_code(self) -> str:
        return _tree2code(self.model, self.columns)


def make_tree(X: pd.DataFrame, y: np.ndarray, params: GlobalParams,
              max_depth=MAX_DEPTH)->list[MLModel]:
    len_y = len(y)
    result = []

    if len_y > DT_MIN_LEN:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=params.seed)
        model.fit(X, y)
        if np.isclose(1, accuracy_score(model.predict(X), y)):
            result.append(DTClassifier(model, list(X.columns)))

    return result


def _tree2code(model: DecisionTreeClassifier,
               param_names: list[str])->str:
    tree_ = model.tree_
    feature_name = [
        param_names[i] if i != UNDEFINED_VALUE else "undefined!"
        for i in tree_.feature
    ]
    ans = []
    _traverse(ans, feature_name, model.classes_, '', 0, tree_)
    return '\n'.join(ans)


def _traverse(result: list[str], feature_name: list[str], classes: list[int],
              indent: str, node: int, tree_)->None:
    if tree_.feature[node] != UNDEFINED_VALUE:
        name = feature_name[node]
        threshold = tree_.threshold[node]
        left_node = tree_.children_left[node]
        right_node = tree_.children_right[node]
        next_indent = indent+'  '

        result.append(f'{indent}if {name} <= {threshold}:')
        _traverse(result, feature_name, classes, next_indent, left_node, tree_)
        result.append(f'{indent}else:')
        _traverse(result, feature_name, classes, next_indent, right_node, tree_)
    else:
        value = tree_.value[node]
        class_ = classes[np.argmax(value[0])]
        result.append(f'{indent}  return {class_}')
