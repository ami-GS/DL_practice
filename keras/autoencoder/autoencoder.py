from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU

import sys
from keras.datasets import mnist
from keras.optimizers import Adam, Adadelta
import matplotlib.pyplot as plt

def encoder_model():
    model = Sequential()
    model.add(Dense(input_dim=784, units=392))
    model.add(LeakyReLU(0.2))
    model.add(Dense(units=196))
    model.add(LeakyReLU(0.2))
    return model

def decoder_model():
    model = Sequential()
    model.add(Dense(input_dim=196, units=392))
    model.add(LeakyReLU(0.2))
    model.add(Dense(units=784))
    model.add(LeakyReLU(0.2))
    return model

BATCH_SIZE=10
NUM_EPOCH=10

def train():
    (X_train, _), (X_test, _) = mnist.load_data()
    shape1 = X_train.shape
    X_train = X_train.astype('float32') / 255.0
    X_train = X_train.reshape((shape1[0], shape1[1]*shape1[2]))
    shape2 = X_test.shape
    X_test = X_test.reshape((shape2[0], shape2[1]*shape2[2]))

    encoder = encoder_model()
    decoder = decoder_model()

    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    autoencoder.fit(X_train, X_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, validation_data=(X_test, X_test))

    autoencoder.save_weights('autoencoder.h5')

def predict():
    encoder = encoder_model()
    decoder = decoder_model()
    (_, _), (X_test, _) = mnist.load_data()
    X_test = X_test.astype('float32') / 255.0
    shape = X_test.shape
    X_test = X_test.reshape((shape[0], shape[1]*shape[2]))

    autoencoder = Sequential([encoder, decoder])    
    autoencoder.load_weights('autoencoder.h5')

    decoded_imgs = autoencoder.predict(X_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    elif  sys.argv[1] == 'predict':
        predict()