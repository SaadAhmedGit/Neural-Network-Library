from Layer import Layer, np
from activations import identity, sigmoid, tanh, relu, softmax


class Activation(Layer):
    def __init__(self, activation_name):
        all_activations = {
            "identity": identity,
            "sigmoid": sigmoid,
            "tanh": tanh,
            "relu": relu,
            "softmax": softmax,
        }
        try:
            self.activation = all_activations[activation_name]
        except KeyError:
            raise ValueError("activation function not defined")

    def forward(self, Z):
        self.input = Z
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * self.activation(self.input, True)
