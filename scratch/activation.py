import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self):
        super(Activation, self).__init__()

class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # destructive assignment
        self.Y = np.reciprocal(1 + np.exp(-x))

        return self.Y

    def backward(self, err_delta, learning_rate):
        self.E = err_delta
        err_delta = np.multiply(err_delta, np.multiply(self.Y, 1-self.Y))
        return err_delta

class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        # destructive assignment
        self.Y = np.multiply(np.greater_equal(x, 0), x)

        return self.Y

    def backward(self, err_delta, learning_rate):
        self.E = err_delta
        err_delta *= np.greater(self.Y, 0)*1

        return err_delta

class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        # destructive assignment
        self.Y = np.tanh(x)

        return self.Y

    def backward(self, err_delta, learning_rate):
        self.E = err_delta
        np.multiply(err_delta, 1 - np.power(self.Y, 2), err_delta)

        return err_delta