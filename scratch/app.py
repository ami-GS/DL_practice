from layer import FullyConnect, Conv2D, MaxPooling2D
from activation import Sigmoid, ReLU, Tanh, Softmax
from network import Network
from loss import MSE
import numpy as np

if __name__ == "__main__":
    last_units = 2
    epoch = 1000
    dataNum = 100000
    batchSize = 10

    # 8*8 image
    input_shape = 64
    dataset = np.zeros((dataNum, input_shape))
    dataset[:dataNum/2,:] = np.random.uniform(-1, -2, (dataNum/2,input_shape))
    dataset[dataNum/2:,:] = np.random.uniform(1, 2, (dataNum/2,input_shape))
    
    label = np.zeros((dataNum, last_units))
    label[:dataNum/2,:] = np.ones((dataNum/2, last_units))

    """
    net = Network(
        [Conv2D(4, 1, 3, input_shape=input_shape),
         Softmax(),
         MaxPooling2D(2),
         FullyConnect(units=last_units, input_shape=100)]) # 5*5* 4filter
    """

    net = Network(
        [FullyConnect(input_shape*2, input_shape),
         ReLU(),
         Softmax(),
         FullyConnect(units=last_units, input_shape=input_shape*2)], batch = batchSize)

    for e in range(epoch):
        for batchIdx in range(0, batchSize, dataNum):
            batchData = dataset[batchIdx:batchIdx + batchSize]
            batchLabel = label[batchIdx:batchIdx + batchSize]
            net.train(batchData, batchLabel)
            err = net.train(batchData, batchLabel, loss=MSE(), learning_rate=0.02)
            if e%100 == 0:
                print "epoch", e
                print "\t", err
    
    #for x in dataset:
        #    print net.predict(x)
    print net.predict(dataset[0, :]), label[0,:]
    print net.predict(dataset[dataNum-1, :]), label[dataNum-1,:]


                