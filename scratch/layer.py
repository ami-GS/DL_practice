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


class Conv2D(Layer):
    def __init__(self, filters, channel, kernel_size, strides=(1,1), input_shape = None):
        super(FullyConnect, self).__init__(input_shape, None)
        self.filterNum = filters
        self.kernel_size = kernel_size
        self.filters = np.random.uniform(-1, 1, (channel, filters, kernel_size, kernel_size))
        self.channel = 1 # WIP
        self.strides = strides
        # size is valid
        self.units = int(np.sqrt(self.input_shape)) - self.kernel_size + 1
        self.units *= self.units

    def forward(self, x):
        self.X = x
         if x.shape >= 2:
            self.channel = x.shape[0]

        x_rowcol = int(np.sqrt(self.input_shape))
        y_rowcol = x_rowcol - self.kernel_size+1

        self.X = np.reshape(X, (channel, x_rowcol, x_rowcol))
        self.Y = np.zeros((self.filterNum, y_rowcol, y_rowcol))
        for f in range(self.filterNum):
            for c in range(channel):
                for xi in range(x_rowcol-self.kernel_size+1):
                    for xj in range(x_rowcol-self.kernel_size+1):
                        self.Y[f][xi][xj] = np.sum(np.multiply(X[c,xi:xi+self.kernel_size,xj:xj+self.kernel_size], self.filters[f,:,:]))

        return self.Y

    def backword(self, err_delta, learning_rate):
        #TBD
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