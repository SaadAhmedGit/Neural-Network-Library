from Layer import Layer, np
from Activation_Layer import Activation


class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, **kwargs):
        self.W = np.random.randn(n_neurons, n_inputs)
        self.B = np.random.randn(n_neurons, 1)
        activation_name = kwargs.get("activation")
        self.activation_layer = Activation(activation_name)

    def forward(self, X):
        self.input = X
        self.output = np.matmul(self.W, self.input) + self.B
        return self.activation_layer.forward(self.output)

    def backward(self, output_delta):
        output_delta = self.activation_layer.backward(output_delta)
        self.dW += np.dot(np.atleast_2d(output_delta), np.atleast_2d(self.input).T)
        self.db += output_delta
        delta = np.dot(self.W.T, output_delta)

        return delta

    def update_params(self, lr):
        self.W -= lr * self.dW
        self.B -= lr * self.db

    def init_gradients(self):
        self.dW = np.zeros((self.W.shape), dtype=float)
        self.db = np.zeros((self.B.shape), dtype=float)
