from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

model = Sequential()
model.add(Conv2D(96, kernel_size=(11, 11), activation='relu', strides=4, input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
model.add(ZeroPadding2D(padding=(2,2)))
model.add(Conv2D(256, kernel_size=(5,5), strides=1))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(384, kernel_size=(3,3), strides=1))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, kernel_size=(3,3), strides=1))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
model.add(Dense(4096))
model.add(Dense(4096))
model.add(Dense(1000))

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metric=['accuracy'])


