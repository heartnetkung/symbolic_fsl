from ..arc.base import *
import numpy as np


def random_with_seed(size: int, seed: int = 0) -> np.ndarray:
    return GlobalParams(seed=seed).random(size)

