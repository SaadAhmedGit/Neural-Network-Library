import numpy as np
from numba import jit


@jit(nopython=True)
def identity(Z, derivative=False):
    return Z


@jit(nopython=True)
def relu(Z, derivative=False):
    if derivative:
        return Z > 0
    return np.maximum(0, Z)


@jit(nopython=True)
def tanh(Z, derivative=False):
    if derivative:
        return np.power(1 - np.tanh(Z), 2)
    return np.tanh(Z)


@jit(nopython=True)
def sigmoid(Z, derivative=False):
    if derivative:
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)
    return 1 / (1 + np.exp(-Z))


@jit(nopython=True)
def softmax(Z, derivative=False):
    if derivative:
        s = Z.reshape(-1, 1)
        temp = np.diagflat(s) - np.dot(s, s.T)
        temp2 = np.diagonal(temp).copy()
        return np.atleast_2d(temp2[temp2 != 0]).T

    eZ = np.exp(Z)
    return eZ / np.sum(eZ)
