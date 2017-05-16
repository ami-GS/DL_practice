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
        self.Y = np.zeros(self.units)
        for i in range(self.units):
            self.Y[i] =  np.sum(np.dot(x, self.W[:,i]))
        self.Y += self.bias
        return self.Y

    def backward(self, err_delta, learning_rate):
        self.E = err_delta
        err_delta = np.zeros(self.input_shape)
        for i in range(self.input_shape):
            err_delta[i] = np.sum(np.dot(self.E, self.W[i,:]))

        for i in range(self.units):
            self.W[:,i] -= np.sum(np.multiply(learning_rate * self.E[i], self.X))

        return err_delta