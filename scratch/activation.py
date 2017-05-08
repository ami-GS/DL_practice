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

    def backward(self):
        pass