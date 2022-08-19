import numpy as np
from losses import *
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        pass

    def Sequential(self, **kwargs):
        layers_list = kwargs.get("layers")
        self.network_layers = np.array(layers_list)

    def feed_forward(self, X):
        curr_input = X
        for curr_layer in self.network_layers:
            curr_input = curr_layer.forward(curr_input)

    def predict(self, X):
        self.feed_forward(X)
        prediction_vector = np.where(self.network_layers[-1].output > 0.5, 1, 0)
        print(f"Output: {prediction_vector}")

    def evaluate(self, X_test, Y_test):
        m = X_test.shape[0]
        correct = 0
        for x, y in zip(X_test, Y_test):
            self.feed_forward(x)
            prediction_vector = np.where(self.network_layers[-1].output > 0.5, 1, 0)
            y = np.reshape(y, (y.shape[0], 1))
            correct += np.array_equal(prediction_vector, y)

        print(f"Model accuracy: {correct*100.0/m}")

    def compile(self, **kwargs):
        self.lr = kwargs.get("lr")
        loss_string = kwargs.get("loss")
        if loss_string == "bce":
            self.loss_function = binary_crossentropy
        elif loss_string == "mse":
            self.loss_function = mean_squared_error
        else:
            raise Exception(f"Loss function '{loss_string}' is not defined")

    def Train(self, X_train, Y_train, **kwargs):
        epochs = kwargs.get("epochs")
        verbose = kwargs.get("verbose")

        if verbose:
            loss_values = []
        for epoch in range(epochs):
            loss = 0
            for x, y in zip(X_train, Y_train):
                self.feed_forward(x)
                output = self.network_layers[-1].activation_layer.output
                y = np.reshape(y, (y.shape[0], 1))
                gradient = self.loss_function(output, y, True)

                for layer in self.network_layers[-1:0:-1]:
                    gradient = layer.backward(gradient, self.lr)

                loss += np.sum(self.loss_function(output, y)).item()

            loss /= X_train.shape[0]

            if verbose:
                print(f"{epoch+1}/{epochs} - loss: {loss}")
                loss_values.append(loss)

        if verbose:
            plt.plot(loss_values)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.show()
