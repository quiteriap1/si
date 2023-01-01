import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the cross-entropy loss function.
    It's calculated by the difference of two probabilities, the true values and the predicted ones.
    *Entropy is the information of a message, in this context, of a variable.
    :param y_true: The true labels of the dataset.
    :param y_pred: The predicted labels of the dataset.
    :return: The cross-entropy loss function.
    """
    return - np.sum(y_true) * np.log(y_pred) / len(y_true)


#Exercise 11.1_Adiciona uma nova medida de erro chamada cross entropy
#Exercise 11.2_Adiciona agora a derivada da medida de erro cross entropy

def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the derivative of the cross-entropy loss function.
    :param y_true: The true labels of the dataset.
    :param y_pred: The predicted labels of the dataset.
    :return: The derivative of the cross-entropy loss function
    """

    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)