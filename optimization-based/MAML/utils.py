import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def cross_entropy(y, y_hat):
    return - np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))/y.shape[0]