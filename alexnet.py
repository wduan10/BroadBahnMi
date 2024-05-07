import os 
import datetime 
import pandas as pd
from IPython.display import display
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras 
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import Model, Sequential
from keras import layers 
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
from keras.src.applications.densenet import DenseNet121, preprocess_input 
from keras.src.applications.resnet import ResNet50, preprocess_input
from keras.src.applications.vgg16 import VGG16, preprocess_input

# AlexNet
model = Sequential()

# Layer 1: Convolutional layer with 64 filters of size 11x11x3
model.add(Conv2D(filters=64, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu', input_shape=(224,224,3)))

# Layer 2: Max pooling layer with pool size of 3x3
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
model.add(Conv2D(filters=192, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# Layer 6: Fully connected layer with 4096 neurons
model.add(Flatten())
model.add(Dense(4096, activation='relu'))

# Layer 7: Fully connected layer with 4096 neurons
model.add(Dense(4096, activation='relu'))