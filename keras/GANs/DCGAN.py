from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

import math
import numpy as np

import os
from keras.datasets import mnist
from keras.optimizers import Adam
from PIL import Image

def generator_model():
    model = Sequential()
    # in 100
    model.add(Dense(input_dim=100, units=1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((7,7,128), input_shape=(128*7*7,)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64,(5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(1,(5,5), padding='same'))
    model.add(Activation('tanh'))
    # out 28 28 1
    return model


def discriminator_model():
    model = Sequential()
    # in 28 28 1
    model.add(Conv2D(64,(5,5), padding='same', strides=(2,2), input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5,5), strides=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # out 1
    return model
    

def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = int(math.ceil(float(total)/cols))
    width, height = generated_images.shape[1:-1]
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image


BATCH_SIZE=32
NUM_EPOCH=20
GENERATED_IMAGE_PATH="generated_images/"

def train():
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train= (X_train.astype(np.float32) - 127.5)/127.5
    X_train= X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1], 1)
    discriminator = discriminator_model()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5, beta_1=0.1))

    discriminator.trainable = False
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    dcgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-4, beta_1=0.5))

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print("Number of batches:", num_batches)
    for epoch in range(NUM_EPOCH):
        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            if index % 500 == 0:
                image = combine_images(generated_images)
                image = image*127.5 + 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8)).save(GENERATED_IMAGE_PATH+"%04d_%04d.png" % (epoch, index))

            X = np.concatenate((image_batch, generated_images))

            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))
            generator.save_weights('generator.h5')
            discriminator.save_weights('discriminator.h5')
            

if __name__ == "__main__":
    train()