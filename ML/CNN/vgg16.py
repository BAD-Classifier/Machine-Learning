from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class VGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		# print("Height shape: " + str(inputShape[0])))
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			input_shape = (depth, height, width)
			chanDim = 1

        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same',))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1000, activation='softmax'))

        return model

