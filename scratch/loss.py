from layer import Layer
import numpy as np

class Loss(object):
    def __init__(self, input_shape=None):
        pass

    def calc(self):
        pass

    def partial_derivative(self):
        pass

class MSE(Loss):
    def __init__(self, input_shape=None):
        super(MSE, self).__init__(input_shape)

    def calc(self, X, label):
        if len(X.shape) != 1 or X.shape != label.shape:
            print "loss error"

        Y = np.zeros(X.shape)
        for i in range(self.input_shape):
            Y[i] = np.power(np.abs(X[i] - label[i]), 2)

        return np.sum(Y)*0.5


        return np.sum(self.Y)/2
            