from layer import Layer
import numpy as np

class Loss(Layer):
    def __init__(self, input_shape=None):
        super(Loss, self).__init__(input_shape)

    def calc(self):
        pass

class MSE(Loss):
    def __init__(self, input_shape=None):
        super(MSE, self).__init__(input_shape)

    def calc(self, X, label):
        if len(X.shape) != 1 or X.shape != label.shape:
            print "loss error"

        for i in range(self.input_shape):
            self.Y[i] = np.power(np.abs(X[i] - label[i]), 2)

        return np.sum(self.Y)/2
            