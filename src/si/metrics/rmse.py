
import pandas as pd
import numpy as np
import sys


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(y_true-y_pred, 2)) / len(y_true))