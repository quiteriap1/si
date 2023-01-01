from typing import Callable

import numpy as np

from data.dataset import Dataset
from metrics.accuracy import accuracy
from metrics.mse import mse, mse_derivative


class NN:
    def __init__(self, layers: list):
        # parameters
        self.layers = layers

    def fit(self, dataset: Dataset) -> "NN":
        """
        Trains the neural network.
        :param dataset: dataset to train the neural network
        :return: Returns the trained neural network.
        """
        x = dataset.X
        for layer in self.layers:
            x = layer.forward(x)


        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the classes of the dataset.
        :param dataset: dataset to predict the classes
        :return: Returns the predicted classes.
        """

        x = dataset.X

        # forward propagation
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def score(self, dataset: Dataset, score_func: Callable = accuracy) -> float:
        """
        Returns the accuracy of the model.
        :param dataset: Dataset object.
        :param score_func: Function to calculate the score.
        :return: Accuracy.
        """
        predictions = self.predict(dataset)

        return score_func(dataset.y, predictions)


class Backpropagation:
    def __init__(self,
                 layers: list,
                 epochs: int = 1000,
                 learning_rate: float = 0.01,
                 loss: Callable = mse,
                 loss_derivative: Callable = mse_derivative,
                 verbose: bool = False):

        """
        Initializes the backpropagation algorithm.
        :param layers: layers of the neural network
        :param epochs: number of epochs
        :param learning_rate: learning rate
        :param loss: loss function
        :param loss_derivative: loss derivative function
        :param verbose: if True will print the loss of each epoch
        """

        # parameters
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose

        # attributes
        self.history = {}

    def fit(self, dataset: Dataset) -> 'Backpropagation':
        """
        Trains the neural network.
        :param dataset: dataset to train the neural network
        :return: Returns the trained neural network.
        """

        for epoch in range(1, self.epochs + 1):

            # forward propagation

            # Extract the input data and the target data
            y_pred = np.array(dataset.x)
            y_true = np.reshape(dataset.y, (-1, 1))

            for layer in self.layers:
                y_pred = layer.forward(y_pred)

            # backward propagation
            
            error = self.loss_derivative(y_true, y_pred)  # y predicted, y real

            # the error is propagated backwards
            for layer in self.layers[::-1]:
                # the list is reversed to propagate the error backwards, we start by the last layer
                error = layer.backward(error, self.learning_rate)

            # saves history cost
            cost = self.loss(y_true, y_pred)  # now with mse
            self.history[epoch] = cost

            # prints the loss value if verbose is True
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} --- cost: {cost}')

            return self