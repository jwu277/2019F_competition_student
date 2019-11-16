import math
import numpy as np
import re
import os
import cv2

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image

#for training the CNN
from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend

from ipywidgets import interact
import ipywidgets as ipywidgets

path = os.getcwd() + "/cropped pictures"
#create the datasets
dataset_orig = np.array([[np.array(Image.open(path + "/" + filename)), filename] \
                    for filename in os.listdir(path)])

#Generate X and Y datasets
X_dataset_orig = np.array([data[0] for data in dataset_orig])
Y_dataset_orig = np.array([data[1][0] for data in dataset_orig]).T

#turn all chars into ints
Y_dataset_orig = np.array([ord(data) for data in Y_dataset_orig])

#make the ints range from 0-35, 0-9 numbers and 10-35 letters
for data in Y_dataset_orig:
    index = np.where(Y_dataset_orig == data)[0][0]
    
    if data <= 57:
        Y_dataset_orig[index] = data - 48
    else:
        Y_dataset_orig[index] = data - 55

#Normalize x (images) dataset
X_dataset = X_dataset_orig/255.

NUMBER_OF_LABELS = 36

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#Convert Y dataset to one-hot encoding
Y_dataset = convert_to_one_hot(Y_dataset_orig, NUMBER_OF_LABELS).T

VALIDATION_SPLIT = 0.2

#Train CNN 
def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                             input_shape=(100, 100, 1)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(36, activation='softmax'))

conv_model.summary()

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

reset_weights(conv_model)

X_dataset = X_dataset[:,:,:,np.newaxis]
# print(X_dataset)
# print(Y_dataset)

history_conv = conv_model.fit(X_dataset, Y_dataset, 
                              validation_split=VALIDATION_SPLIT, 
                              epochs=20, 
                              batch_size=16)

#plot the models history
plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()