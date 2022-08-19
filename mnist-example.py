from keras.datasets import mnist
from Model import Model
import numpy as np
from Flatten_Layer import Flatten
from Dense_Layer import Dense

# Loading the dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Scaling the values
X_train, X_test = X_train / 255.0, X_test / 255.0


def to_categorical(Y, num_classes):
    return np.eye(num_classes)[Y.astype(int)]


Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

myModel = Model()
myModel.compile(loss="mse", lr=0.01)

myModel.Sequential(
    layers=[
        Flatten(input_shape=(28, 28)),
        Dense(28 * 28, 128, activation="sigmoid"),
        Dense(128, 10, activation="sigmoid"),
    ]
)

myModel.Train(X_train, Y_train, epochs=5, batch_size=1, verbose=True)
myModel.evaluate(X_test, Y_test)
