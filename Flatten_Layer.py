from Layer import Layer, np
from Activation_Layer import Activation


class Flatten(Layer):
    def __init__(self, **kwargs):
        self.input_shape = kwargs.get("input_shape")
        self.activation_layer = Activation("identity")

    def forward(self, X):
        self.input = X
        self.output = np.resize(X, (np.product(self.input_shape), 1))
        return self.activation_layer.forward(self.output)
