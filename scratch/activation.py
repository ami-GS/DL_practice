import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self):
        super(Activation, self).__init__()

class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        for j in range(self.units):
            self.Y[j] = 1.0/(1.0 + np.exp(-x[j]))

        return self.Y

    def backward(self, err_delta, learning_rate):
        self.E = err_delta
        for i in range(self.units):
            err_delta[i] *= self.Y[i] * (1 - self.Y[i])
        return err_delta

class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        for j in range(self.units):
            self.Y[j] = max(0, x[j])

        return self.Y

    def backward(self, err_delta, learning_rate):
        self.E = err_delta
        for i in range(self.units):
            if self.Y[i] > 0:
                err_delta[i] *= 1
            elif self.Y[i] < 0:
                err_delta[i] = 0
            else:
                err_delta[i] *= np.random.uniform(0,1,1)

        return err_delta

class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        for i in range(self.units):
            self.Y[i] = np.tanh(x[i])

        return self.Y

    def backward(self, err_delta, learning_rate):
        for i in range(self.units):
            err_delta[i] *= 1-self.Y[i]**2

        return err_delta