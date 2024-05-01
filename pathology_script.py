import os, sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras 
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import Model
from keras import layers 
from sklearn.model_selection import train_test_split
from keras.src.applications.vgg16 import VGG16, preprocess_input

hpc = False
print(sys.argv)
if (len(sys.argv) > 1 and sys.argv[1] == 'hpc'):
    hpc = True

if (hpc):
    labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv'
    labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv'
    img_dir = '/groups/CS156b/data'

    df_train = pd.read_csv(labels_path_train)[:-1]
else:
    labels_path_train = 'data/train/labels/labels.csv'
    labels_path_test = 'data/test/ids.csv'
    img_dir = 'data'

    df_train = pd.read_csv(labels_path_train)

BATCH_SIZE = 64
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print()
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

df_test = pd.read_csv(labels_path_test)
print(df_train)
print(df_test)

def parse_labels(df):
    df.fillna(0, inplace=True)
    return df

classes = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
           "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

df = pd.DataFrame()
pathology = 'Fracture'
df['filename'] = df_train['Path']
df['label'] = df_train[pathology]

if (gpus):
    df['label'] = df['label'][:-1]

# remove Nan values
df = df.dropna()

# 'categorical' requires strings
df['label'] = df['label'].astype(str)

# Stratified train/test split based on 'Frontal/Lateral' column
train_df, val_df = train_test_split(df,
                                    test_size=0.5,
                                    random_state=42)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rescale=1./255, #Normalize
                                   zoom_range=0.4,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 rescale=1./255)

# Apply the ImageDataGenerator to create image batches
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=img_dir,
    x_col='filename',
    y_col='label',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    target_size=(224, 224),
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=img_dir,
    x_col='filename',
    y_col='label',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    target_size=(224, 224),
)

# VGG16 Model
conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Customize top layer
top_layer = conv_base.output
top_layer = keras.layers.GlobalAveragePooling2D()(top_layer)
top_layer = keras.layers.Dense(4096, activation='relu')(top_layer)
top_layer = keras.layers.Dense(1072, activation='relu')(top_layer)
top_layer = keras.layers.Dropout(0.2)(top_layer)
output_layer = keras.layers.Dense(2, activation='softmax')(top_layer) # Predicting for one pathology

model = Model(inputs=conv_base.input, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=1,
    validation_data=val_generator,
    verbose=1)

model.evaluate(val_generator)