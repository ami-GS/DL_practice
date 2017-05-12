from layer import FullyConnect
from activation import Sigmoid
from network import Network
from loss import MSE
import numpy as np

if __name__ == "__main__":
    input_shape = 2
    last_units = 1
    epoch = 200000

    # XOR
    dataset = np.asarray([[0,0], [0,1], [1,0], [1,1]])
    label = np.asarray([[0],[1],[1],[0]])
    
    net = Network(
        [FullyConnect(units=32, input_shape=input_shape),
         Sigmoid(),
         FullyConnect(units=last_units, input_shape=32)])

    for e in range(epoch):
        if e ==0 or e == (epoch-1):
            print "epoch", e
        for i in range(len(dataset)):
            err = net.train(dataset[i], label[i], loss=MSE(), learning_rate=0.02)
            if e ==0 or e == (epoch-1):
                print "\t", err

    for x in dataset:
        print net.predict(x)