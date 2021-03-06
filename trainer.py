from keras import backend as K
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt

# Load Data from folder 
current_path = os.getcwd()
All_files = glob.glob(current_path + '/Reduced_Data_further/**/*')
print("Number of total files: " + str(len(All_files)))

######--HYPERPARAMETERS--#######
Epochs = 1000             
Learning_Rate = 1e-5            
Batch_Size = 5                    
Image_Dimensions = (120, 84, 3)
model_name = '2-32-2-64-2-128--5'
act_type = "relu"  
filter_size = (3, 3)
pad_type = "same"
dropout_rate = 0.5
################################

# data and label intitialization
os.makedirs(model_name)
data = []
labels = []

random.shuffle(All_files)

for image_name in All_files:
    image_data = cv2.imread(image_name)
    image_data = cv2.resize(image_data, (Image_Dimensions[0], Image_Dimensions[1]))
    image_data = img_to_array(image_data)
    data.append(image_data)
    label = image_name.split(os.path.sep)
    labels.append(label[-2])

data = np.array(data, dtype="float") /255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Data split into training and testing set
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2)

# Data Augmentation
data_augmentation = ImageDataGenerator(width_shift_range=0.15, height_shift_range=0.15)

# Model inputs
width = Image_Dimensions[0]
height = Image_Dimensions[1]
depth = Image_Dimensions[2]
input_parameters = (height, width, depth)
channel_dimensions = -1
classes = len(lb.classes_)

# Channels depend on the OS used
if K.image_data_format() == "channels_first":
    input_parameters = (depth, height, width)
    channel_dimensions = 1

# Defining model architecture
model = Sequential()
model.add(Conv2D(32, filter_size, padding=pad_type, input_shape=input_parameters))
model.add(Activation(act_type))
model.add(BatchNormalization(axis=channel_dimensions))
model.add(Dropout(dropout_rate))

model.add(Conv2D(32, filter_size, padding=pad_type))
model.add(Activation(act_type))
model.add(BatchNormalization(axis=channel_dimensions))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(dropout_rate))

model.add(Conv2D(64, filter_size, padding=pad_type))
model.add(Activation(act_type))
model.add(BatchNormalization(axis=channel_dimensions))
model.add(Dropout(dropout_rate))

model.add(Conv2D(64, filter_size, padding=pad_type))
model.add(Activation(act_type))
model.add(BatchNormalization(axis=channel_dimensions))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_rate))

model.add(Conv2D(128, filter_size, padding=pad_type))
model.add(Activation(act_type))
model.add(BatchNormalization(axis=channel_dimensions))
model.add(Dropout(dropout_rate))

model.add(Conv2D(128, filter_size, padding=pad_type))
model.add(Activation(act_type))
model.add(BatchNormalization(axis=channel_dimensions))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(1024, use_bias=False))
model.add(BatchNormalization())
model.add(Activation(act_type))
model.add(Dropout(dropout_rate+0.3))

model.add(Dense(classes))
model.add(Activation("softmax"))

# Model compilation
opt = Adam(lr=Learning_Rate, decay=Learning_Rate/Epochs)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
filepath = model_name + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train Network
model_history = model.fit_generator(data_augmentation.flow(x_train, y_train, batch_size=Batch_Size), validation_data=(x_test, y_test),
   steps_per_epoch=(len(x_train)//Batch_Size), epochs=Epochs, verbose=1, callbacks=callbacks_list)
data_augmentation.flow(x_train, y_train, batch_size=Batch_Size)

# Save Model 
model.save(model_name+ '/' + model_name+'.model')

# Save Labels
pickle_file = open('labels.pickle', "wb")
pickle_file.write(pickle.dumps(lb))
pickle_file.close()

# Plot loss and accuracy of model
plt.figure()
plt.plot(np.arange(0, Epochs), model_history.history["acc"], label="Training Accuracy")
plt.plot(np.arange(0, Epochs), model_history.history["val_acc"], label="Validation Accuracy")
plt.title("Accuracy vs Validation Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.savefig(model_name+ '/' + model_name + '-acc.png')

plt.figure()
plt.plot(np.arange(0, Epochs), model_history.history["loss"], label="Training Loss")
plt.plot(np.arange(0, Epochs), model_history.history["val_loss"], label="Validation Loss")
plt.title("Loss vs Validation Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.savefig(model_name+ '/' + model_name + '-loss.png')

# Creating arrays of the data to plot manually 
accuracy_history = np.array(model_history.history["acc"])
val_accuracy_history = np.array(model_history.history["val_acc"])
loss_history = np.array(model_history.history["loss"])
val_loss_history = np.array(model_history.history["val_loss"])

# Saving arrays to text files
np.savetxt(model_name+ '/' + model_name + "-accuracy.txt", accuracy_history, delimiter=",")
np.savetxt( model_name+ '/' + model_name + "-val_accuracy.txt", val_accuracy_history, delimiter=",")
np.savetxt(model_name+ '/' + model_name + "-loss.txt", loss_history, delimiter=",")
np.savetxt(model_name+ '/' + model_name + "-val_loss.txt", val_loss_history, delimiter=",")
