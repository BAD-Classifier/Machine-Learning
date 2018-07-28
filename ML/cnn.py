

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
from keras.preprocessing import image
import matplotlib.pylab as plt

import numpy as np
import cv2
from numpy import array


batch_size = 5
num_classes = 2
epochs = 10


# input image dimensions
img_x, img_y = 1400, 400

# load the MNIST data set, which already splits into train and test sets for us
 
path = os.getcwd() + '/Train/'
fileNames = os.listdir(path)

y_train = []
x_train = []
y_test = []
x_test = []

for fileName in fileNames:
    w, x, _, _ = fileName.split('-')
    name = w + " " + x
    if name == "Telophorus zeylonus":
        y_train.append(0)
    else:
        y_train.append(1)
    
    im = cv2.imread(path + fileName)
    x_train.append(im)


print("Number of train variables: " + str(len(x_train)))
print("Number of train labels: " + str(len(y_train)))

path = os.getcwd() + '/Test/'
fileNames = os.listdir(path)

for fileName in fileNames:
    w, x, _, _ = fileName.split('-')
    name = w + " " + x
    im_test = cv2.imread(path + fileName)
    x_test.append(im_test)
    #y_test.append(name)
    if name == "Telophorus zeylonus":
        y_test.append(0)
    else:
        y_test.append(1)
    
print("Number of test variables: " + str(len(x_test)))
print("Number of test labels: " + str(len(y_test)))

# # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = array( x_train )
x_test = array( x_test )

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)

print("x train set shape")
print(x_train.shape)
print("x test set shape")
print(x_test.shape)

input_shape = (img_x, img_y, 3)
print("Input shape:")
print(input_shape)


# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255   #due to pixel being able to take 1-255 values, normalization
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, 2) 
y_test = keras.utils.to_categorical(y_test, 2)

# need one hot encoding
print(type(y_train)) #(107,2)
print(y_test) #(0,2) 

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
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
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
\
