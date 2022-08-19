import numpy as np

# @jit(nopython=True)
def mean_squared_error(Y, Y_hat, derivative=False):
    m = Y_hat.shape[0]
    if derivative:
        return (Y - Y_hat) / m
    return (np.power(Y - Y_hat, 2)) / (2 * m)


# @jit(nopython=True)
def binary_crossentropy(Y, Y_hat, derivative=False):
    m = Y.shape[0]
    epsilon = 1e-5
    if derivative:
        return ((1 - Y_hat) / (1 - Y + epsilon) - (Y / (Y_hat + epsilon))) / m
    return (
        np.sum(
            -Y_hat * np.log(Y + epsilon)
            - (1 - Y_hat + epsilon) * np.log(1 - Y + epsilon)
        )
        / m
    )
