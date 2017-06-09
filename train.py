import coremltools
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Reshape, Permute
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('th')

images_input_shape = (3, 100, 100)


model = Sequential()

model.add(Convolution2D(32, 3, 3,border_mode='same', input_shape=images_input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print model.input_shape

model.load_weights("weights.hdf5")
# model.add(Permute((2, 1, 3), input_shape=(100, 3, 100)))

# model.add(Reshape((100, 100, 3), input_shape=images_input_shape))


print model.input_shape

coremlmodel = coremltools.converters.keras.convert(model, class_labels="labels.txt", image_input_names="input1")


# coremlmodel.save("Keras.mlmodel")