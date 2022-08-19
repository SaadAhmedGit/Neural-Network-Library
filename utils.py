import numpy as np

def to_categorical(Y, num_classes):
    return np.eye(num_classes)[Y.astype(int)]
