import numpy as np

class Layer(object):
    def __init__(self, input_shape = None, units = None):
        self.input_shape = input_shape
        self.units = units
        # input
        self.X = np.zeros(input_shape)
        # calculated value
        self.Y = np.zeros(units)
        # error
        self.E = np.zeros(units)

    def forward(self, x):
        pass

    def backward(self, err_delta, learning_rate):
        pass


class FullyConnect(Layer):
    def __init__(self, units, input_shape):
        super(FullyConnect, self).__init__(input_shape, units)
        self.W = np.random.uniform(-1, 1, (input_shape, units))
        self.bias = np.random.uniform(-1, 1, 1)


    def forward(self, x):
        self.X = x
        self.Y = x.dot(self.W)
        self.Y += self.bias
        return self.Y

    def backward(self, err_delta, learning_rate):
        self.E = err_delta
        err_delta = self.E.dot(self.W.T)
        np.subtract(self.W, np.outer(self.X, learning_rate * self.E), self.W)

        return err_delta