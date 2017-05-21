import numpy as np
from layer import MaxPooling2D, Conv2D
from activation import Activation
from loss import MSE

class Network:
    def __init__(self, layers=None):
        self.layers = layers
        self.input_shape = layers[0].input_shape
        units = self.layers[0].units
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.X = np.zeros(layer.input_shape)
            layer.Y = np.zeros(layer.units)
            layer.E = np.zeros(layer.units)
            if i >= 1:
                layer.input_shape = units
                if isinstance(layer, Activation):
                    layer.units = units
                elif isinstance(layer, MaxPooling2D) or isinstance(layer, Conv2D):
                    tmp = layer.input_shape - layer.kernel_size + 1
                    layer.Y = np.zeros((1, tmp, tmp))
                    layer.units = tmp**2
            units = layer.units
        # current value
        self.Y = None

    def predict(self, X):
        self.Y = X
        for layer in self.layers:
            self.Y = layer.forward(self.Y)
        return self.Y

    def train(self, X, label, loss=MSE(), learning_rate=0.02):
        self.Y = X
        for layer in self.layers:
            self.Y = layer.forward(self.Y)

        err = loss.calc(self.Y, label)
        err_delta = loss.partial_derivative(self.Y, label)
        for layer in self.layers[::-1]:
            err_delta = layer.backward(err_delta, learning_rate)

        return err
