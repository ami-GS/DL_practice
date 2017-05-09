from layer import FullyConnect
from activation import Sigmoid
from network import Network
from loss import MSE
import numpy as np

if __name__ == "__main__":
    input_shape = 1024
    last_units = 10
    epoch = 5
    
    net = Network(
        [FullyConnect(units=512, input_shape=input_shape),
         Sigmoid(),
         FullyConnect(units=last_units, input_shape=512)])

    X = np.random.rand(epoch, input_shape)
    # one-hot expression
    label = np.zeros(last_units)
    label[2] = 1

    for e in range(epoch):
        err = net.train(X[e,:], label, loss=MSE(), learning_rate=0.02)
    print err
