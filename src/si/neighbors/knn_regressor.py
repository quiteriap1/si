# KNNRegressor - estima um valor médio dos k exemplos mais semelhantes
from mimetypes import init
from typing import Callable, Union

import numpy as np

from si.src.si.data.dataset import Dataset
from si.src.si.metrics.rmse import rmse
from si.src.si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:

    def init(self, k:init=1, distance:callable(euclidean_distance)):
        """

        :type distance: object
        """
        # parameters
        self.k = k
        self.distance = distance

        #atributes
        self.dataset=None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:

        #compute the distance between the sample and the dataset

        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors

        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors

        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # calcula a média dos "labels" (valor da variável de interesse) dos k vizinhos mais próximos

        return np.average(k_nearest_neighbors_labels)

    def predict(self, dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)


if __name__== "__main__":
    from src.si.read_csv import read_csv
    dataset_= read_csv("C:\Users\HP-PC\PycharmProjects\pythonProject2\si\datasets\cpu.csv",sep=",")

    knn = KNNRegressor(5)
    knn.fit(dataset_, 0.2)
    score = knn.score(dataset_)
    print(f'The accuracy of the model is: {score}')