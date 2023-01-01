
import numpy as np
from data.dataset import Dataset


class PCA:

    def __init__(self, n_components: int) -> None:
        """
        It performs the Principal Component Analysis (PCA) on a givrn dataset, using the Singular Value Decomposition method.
        Args:
            n_components (int): Number of components to be considered and returned from the analysis.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset) -> "PCA":

        # center data
        self.mean = np.mean(dataset.X, axis=0)
        self.centered_data = dataset.X - dataset.X.mean(axis=0, keepdims=True)

        # SVD  calculate
        U, S, Vt = np.linalg.svd(self.centered_data, full_matrices=False)

        # principal components
        self.components = Vt[:self.n_components]

        # explained variance
        n = len(dataset.X)
        EV = (S**2)/(n-1)
        self.explained_variance = EV[:self.n_components]
        return self

    def transform(self, dataset: Dataset) -> None:
        """
        Returns the calculated reduced Singular Value Decomposition (SVD)
        Args:
            dataset (Dataset): Dataset object
        V corresponde a transposta de Vt (self.components)
        """
        V = self.components.T
        X_reduced = np.dot(self.centered_data, V)
        return X_reduced

    def fit_transform(self, dataset: Dataset) -> None:
        """
        It fit and transform the dataset
        Args:
            dataset (Dataset): Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset=dataset)


if __name__ == "__main__":
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a = PCA(n_components=2)
    print(a.fit_transform(dataset=dataset))