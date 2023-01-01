from typing import Callable

import pandas as pd
import numpy as np

from si.src.si.data.dataset import Dataset
from si.src.si.statistics.f_classification import f_classification


class SelectPercentile:

    def __init__(self, score_func: Callable = f_classification, percentile: int = 10):

        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':

        self.F, self.p = self.score_func(dataset)
        return self

    #ex 3 - aula 2
    def transform(self, dataset: Dataset) -> Dataset:
    #transform – seleciona as features com valor de F mais alto até ao
    #percentil indicado. Por exemplo, para um dataset com 10 features e um
    #percentil de 50%, o teu transform deve selecionar as 5 features com valor
    #de F mais alto

        #self.F = [10, 5, 1]
        # [1,5,10]
        # [2,1,0]
        numero_total_features = len(list(dataset.features))
        print(numero_total_features)
        numero_final = int(numero_total_features * self.percentile) #multiplicar o numero total por metade (self.percentile=0.5)
        print(numero_final)
        idxs = np.argsort(self.F)[-numero_final:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:

        self.fit(dataset)
        return self.transform(dataset)

#aula__2
if __name__ == 'main':
    from si.src.si.data.dataset import Dataset

    df = pd.read_csv("C:\\Users\\HP-PC\\PycharmProjects\\pythonProject2\\si\\datasets\\iris.csv")
    dataset = Dataset.from_dataframe(df, label='class')
    #dataset = Dataset(X=np.array([[0, 2, 0, 3],
    #                               [0, 1, 4, 3],
    #                               [0, 1, 1, 3]]),
    #                   y=np.array([0, 1, 0]),
    #                   features=["f1", "f2", "f3", "f4"],
      #                 label="y")

    selector = SelectPercentile(percentile=0.5)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)

#a resposta deve ser: ['petal_width', 'petal_length']