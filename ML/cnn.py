

from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pylab as plt
import random
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from numpy import array
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from imutils import paths
batch_size = 1
num_classes = 2
epochs = 10


# input image dimensions
img_x, img_y = 1092, 315

# load the MNIST data set, which already splits into train and test sets for us
 
path = os.getcwd() + '/dataset/'
folderNames = sorted(list(os.listdir(path)))
random.seed(42)
random.shuffle(folderNames)
print(folderNames[0])

# y_train = []
# x_train = []
# y_test = []
# x_test = []

data = []
labels = []

for folderName in folderNames:
    folderPath = path + folderName
    fileNames = sorted(list(os.listdir(folderPath)))
    for imageName in fileNames:
        fullPath = folderPath + '/' + imageName
        print(fullPath)
        image = cv2.imread(fullPath)
        image = cv2.resize(image, (img_x, img_y))
        image = img_to_array(image)
        data.append(image)
        label = folderName.split(os.path.sep)
        print(label)
        labels.append(label)


data = np.array(data, dtype="float") /255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)


(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)

print("x train set shape")
print(x_train.shape)
print("x test set shape")
print(x_test.shape)

input_shape = (img_x, img_y, 3)
print("Input shape:")
print(input_shape)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
# y_train = keras.utils.to_categorical(y_train, 2) 
# y_test = keras.utils.to_categorical(y_test, 2)

# # need one hot encoding
# print(type(y_train)) #(107,2)
# print(y_test) #(0,2) 

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,  validation_data=(x_test, y_test),  callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print(model.summary())
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

