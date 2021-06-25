import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def cross_entropy(y_hat, y_true):
    return - np.sum((y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)))/y_true.shape[0]