from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model.add(Conv2D(64, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))
		print("made it this far 3")
		model.add(Conv2D(128, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Conv2D(256, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(256, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# model = Sequential()
		# model.add(Conv2D(64, (3, 3), input_shape=inputShape, padding='same', activation='relu'))
		# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# model.add(Flatten())
		# model.add(Dense(4096, activation='relu'))
		# model.add(Dense(4096, activation='relu'))
		# model.add(Dense(10, activation='softmax'))

		# return the constructed network architecture
		return model