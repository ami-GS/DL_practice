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
        super(Conv2D, self).__init__(input_shape, None)
        self.filterNum = filters
        self.kernel_size = kernel_size
        self.filters = np.random.uniform(-1, 1, (filters, kernel_size, kernel_size))
        self.channel = 1 # WIP
        self.strides = strides
        # size is valid
        self.units = int(np.sqrt(self.input_shape)) - self.kernel_size + 1
        self.units *= self.units

    def forward(self, x):
        self.X = x
        if len(x.shape) >= 2:
            self.channel = x.shape[0]

        self.x_rowcol = int(np.sqrt(self.input_shape))
        self.y_rowcol = self.x_rowcol - self.kernel_size+1

        self.X = np.reshape(self.X, (self.channel, self.x_rowcol, self.x_rowcol))
        self.Y = np.zeros((self.filterNum, self.y_rowcol, self.y_rowcol))
        for f in range(self.filterNum):
            for c in range(self.channel):
                for xi in range(self.x_rowcol-self.kernel_size+1):
                    for xj in range(self.x_rowcol-self.kernel_size+1):
                        self.Y[f,xi,xj] = np.sum(np.multiply(self.X[c,xi:xi+self.kernel_size,xj:xj+self.kernel_size], self.filters[f,:,:]))

        return self.Y

    def backword(self, err_delta, learning_rate):
        #self.E = err_delta
        err_delta = err_delta.reshpae((self.filterNum, self.x_rowcol-self.kernel_size+1, self.x_rowcol-self.kernel_size+1))
        self.E = err_delta.reshpae((self.filterNum, self.x_rowcol-self.kernel_size+1, self.x_rowcol-self.kernel_size+1))

        for xi in range(self.x_rowcol-self.kernel_size+1):
            for xj in range(self.x_rowcol-self.kernel_size+1):
                err_delta[:, xi:xi+self.kernel_size, xj:xj+self.kernel_size] += self.E[:, xi, xj] * self.filters[:,:,:].T

        for f in range(self.filterNum):
            for c in range(self.channel):
                for xi in range(self.x_rowcol-self.kernel_size+1):
                    for xj in range(self.x_rowcol-self.kernel_size+1):
                        self.filters[f,:,:] -= learning_rate * self.X[c, xi:xi+self.kernel_size, xj:xj+self.kernel_size] * self.E[f,xi,xj]

        return err_delta


class FullyConnect(Layer):
    def __init__(self, units, input_shape):
        super(FullyConnect, self).__init__(input_shape, units)
        self.W = np.random.uniform(-1, 1, (input_shape, units))
        self.bias = np.random.uniform(-1, 1, 1)
        self.original_shape = None

    def forward(self, x):
        if len(x.shape) > 1:
            self.original_shape = x.shape
            x = np.ravel(x)
        self.X = x
        self.Y = x.dot(self.W)
        self.Y += self.bias
        return self.Y

    def backward(self, err_delta, learning_rate):
        self.E = err_delta
        err_delta = self.E.dot(self.W.T)
        np.subtract(self.W, np.outer(self.X, learning_rate * self.E), self.W)

        if self.original_shape:
            err_delta = err_delta.reshape(self.original_shape)
        return err_delta