from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile:

    def __init__(self, score_func: Callable = f_classification, percentile: int = 10):

        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':

        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:

        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:

        self.fit(dataset)
        return self.transform(dataset)