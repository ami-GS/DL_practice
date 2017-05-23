from layer import FullyConnect, Conv2D, MaxPooling2D
from activation import Sigmoid, ReLU, Tanh, Softmax
from network import Network
from loss import MSE
import numpy as np

if __name__ == "__main__":
    input_shape = 2
    last_units = 1
    epoch = 5000

    # XOR
    dataset = np.asarray([[0,0], [0,1], [1,0], [1,1]])
    label = np.asarray([[0],[1],[1],[0]])
    
    # 8*8 image
    input_shape = 64
    dataset = np.random.rand(4, input_shape)

    net = Network(
        [Conv2D(4, 1, 3, input_shape=input_shape),
         Softmax(),
         MaxPooling2D(2),
         FullyConnect(units=last_units, input_shape=100)]) # 5*5* 4filter

    for e in range(epoch):
        if e ==0 or e == (epoch-1):
            print "epoch", e
        for i in range(len(dataset)):
            err = net.train(dataset[i], label[i], loss=MSE(), learning_rate=0.02)
            if e ==0 or e == (epoch-1):
                print "\t", err

    for x in dataset:
        print net.predict(x)