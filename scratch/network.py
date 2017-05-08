import numpy as np
from activation import Activation
from loss import MSE

class Network:
    def __init__(self, layers=None):
        self.layers = layers
        self.input_shape = layers[0].input_shape
        units = self.layers[0].units
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i >= 1:
                layer.input_shape = units
                if isinstance(layer, Activation):
                    layer.units = units
            layer.Y = np.zeros(layer.units)
            units = layer.units

        # current value
        self.Y = None

    def train(self, X, label, loss=MSE()):
        self.Y = X
        for layer in self.layers:
            self.Y = layer.forward(self.Y)

        err_delta = loss.partial_derivative(self.Y, label)
        return err_delta
