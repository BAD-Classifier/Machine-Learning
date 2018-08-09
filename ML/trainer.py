import os
import random
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
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense


# Load Data from folder 
dataFolder = '/Cleaned_Data/'
path = os.getcwd() + dataFolder
folderNames = sorted(list(os.listdir(path)))
random.seed(42)
random.shuffle(folderNames)

########--PARAMS--#########
EPOCHS = 5                #
INIT_LR = 1e-3            #
BS = 5                    #
img_x, img_y = 1092, 315  #
IMAGE_DIMS = (200, 100, 3)#
modelName = 'it_9.model'  #
###########################

# Load data and labels into memory
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

print("[INFO] data matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Split Data
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data Augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

# Setup Model
model = Sequential()
width = IMAGE_DIMS[0]
height = IMAGE_DIMS[1]
depth = IMAGE_DIMS[2]
inputShape = (height, width, depth)
chanDim = -1
classes = len(lb.classes_)

if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1

#################################################################################
model.add(Conv2D(64, (5, 5), padding="same", input_shape=inputShape))        
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
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
print(model.summary())
################################################################################

# Optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

# Train Network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1)


# Save Model 
print("[INFO] saving model...")
model.save(modelName)

# Save Labels
print("[INFO] serializing label binarizer...")
f = open('lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(modelName+'.png')