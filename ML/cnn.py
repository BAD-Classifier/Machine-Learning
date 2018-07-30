

from __future__ import print_function

from pyimagesearch.smallervggnet import SmallerVGGNet

import os
import keras
from keras.optimizers import Adam
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
import pickle
import matplotlib.pyplot as plt
from imutils import paths

 
path = os.getcwd() + '/dataset/'
folderNames = sorted(list(os.listdir(path)))
random.seed(42)
random.shuffle(folderNames)
EPOCHS = 1000
INIT_LR = 1e-3
BS = 3
img_x, img_y = 1092, 315
IMAGE_DIMS = (350, 100, 3)

data = []
labels = []

for folderName in folderNames:
    folderPath = path + folderName
    fileNames = sorted(list(os.listdir(folderPath)))
    for imageName in fileNames:
        fullPath = folderPath + '/' + imageName
        image = cv2.imread(fullPath)
        image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
        image = img_to_array(image)
        data.append(image)
        label = folderName.split(os.path.sep)
        labels.append(label)


data = np.array(data, dtype="float") /255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

labels = keras.utils.to_categorical(labels, num_classes=2)

print(labels.shape)

(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)

print("x train set shape")
print(x_train.shape)
print("y train set shape")
print(y_train.shape)
print("x test set shape")
print(x_test.shape)
print("y test set shape")
print(y_test.shape)

input_shape = (img_y, img_x, 3)
print("Input shape:")
print(input_shape)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[0], height=IMAGE_DIMS[1],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print(model.summary())

print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(x_train, y_train, batch_size=BS),
	validation_data=(x_test, y_test),
	steps_per_epoch=len(x_train) // BS,
	epochs=EPOCHS, verbose=1)
# model.fit(x_train, y_train, batch_size=BS, epochs=1, validation_data=(x_test, y_test))

# save the model to disk    
print("[INFO] serializing network...")
model.save('attempt,model')

print("[INFO] serializing label binarizer...")
f = open('lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('plot.png')
