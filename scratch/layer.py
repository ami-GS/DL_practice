import numpy as np

class Layer(object):
    def __init__(self, input_shape = None, units = None):
        self.input_shape = input_shape
        self.units = units
        self.Y = np.zeros(units)

    def forward(self):
        pass

    def backward(self):
        pass


class FullyConnect(Layer):
    def __init__(self, units, input_shape):
        super(FullyConnect, self).__init__(input_shape, units)
        self.W = np.random.uniform(-1, 1, (input_shape, units))


    def forward(self, x):
        for i in range(self.input_shape):
            for j in range(self.units):
                self.Y[j] += x[i]*self.W[i][j]
        return self.Y

    def backword(self):
        pass