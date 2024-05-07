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

BATCH_SIZE = 64 
NUM_EPOCHS = 1 
LEARNING_RATE = 0.0002 
HPC = True 

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

if (HPC):
    labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv'
    labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv'
    img_dir = '/groups/CS156b/data'

    df_train = pd.read_csv(labels_path_train)[:-1]

    TEST_SIZE = 0.2 
else:
    labels_path_train = 'data/train/labels/labels.csv'
    labels_path_test = 'data/test/ids.csv'
    img_dir = 'data'

    df_train = pd.read_csv(labels_path_train)
    TEST_SIZE = 0.5 

df_test = pd.read_csv(labels_path_test)
print(df_train)
print(df_test)

def parse_labels(df):
    df.fillna(0, inplace=True)
    return df

# Prepare data for each Pathology 
def get_pathology(pathology):
    df = pd.DataFrame()
    df['filename'] = df_train['Path']
    df['label'] = df_train[pathology]

    if (HPC):
        df['label'] = parse_labels(df['label'][:-1])
    else:
        df['label'] = parse_labels(df['label'])

    # 'categorical' requires strings
    df['label'] = df['label'].astype(str)

    # Stratified train/test split based on 'Frontal/Lateral' column
    train_df, val_df = train_test_split(df,
                                        test_size=TEST_SIZE,
                                        random_state=42)

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rescale=1./255, #Normalize
                                    zoom_range=0.1,
                                    horizontal_flip=True)
    # train_datagen = ImageDataGenerator(rescale=1./255, #Normalize
    #                                 zoom_range=0.1,
    #                                 horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Apply the ImageDataGenerator to create image batches
    train_data = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=img_dir,
        x_col='filename',
        y_col='label',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        target_size=(224, 224),
    )

    val_data = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=img_dir,
        x_col='filename',
        y_col='label',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        target_size=(224, 224),
    )
    return train_data, val_data

# Handling test data 
test_df = pd.DataFrame()
test_df['filename'] = df_test['Path']
display(test_df.head())

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_dataframe(
    dataframe=test_df,   
    directory=img_dir,
    x_col='filename',
    class_mode=None,   
    batch_size=BATCH_SIZE,
    target_size=(224, 224),
    shuffle=False  
)

classes = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
           "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

pathology = "Fracture"
train_data, val_data = get_pathology(pathology)
display(train_data.head())
display(val_data.head())

class_mapping = train_data.class_indices 
print(class_mapping)

# VGG16 Model
# conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Resnet 
conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# DenseNet
# conv_base = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# AlexNet
# model = Sequential()

# # Layer 1: Convolutional layer with 64 filters of size 11x11x3
# model.add(Conv2D(filters=64, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu', input_shape=(224,224,3)))

# # Layer 2: Max pooling layer with pool size of 3x3
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# # Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
# model.add(Conv2D(filters=192, kernel_size=(5,5), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
# model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# # Layer 6: Fully connected layer with 4096 neurons
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))

# # Layer 7: Fully connected layer with 4096 neurons
# model.add(Dense(4096, activation='relu'))

# Customize top layer
top_layer = conv_base.output
top_layer = keras.layers.GlobalAveragePooling2D()(top_layer)
top_layer = keras.layers.Dense(512, activation='relu')(top_layer)   
top_layer = keras.layers.Dense(128, activation='relu')(top_layer)   
top_layer = keras.layers.Dropout(0.2)(top_layer)
output_layer = keras.layers.Dense(3, activation='softmax')(top_layer) # Predicting for one pathology 

model = Model(inputs=conv_base.input, outputs=output_layer)

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_data,
    epochs=NUM_EPOCHS,
    validation_data=val_data,
    verbose=1)

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
for i in range(NUM_EPOCHS):
    print(f"Epoch {i+1}: Training Loss = {training_loss[i]:.4f}, Validation Loss = {validation_loss[i]:.4f}")

# Dataframe of predictions 
predictions = model.predict(test_data)
preds = pd.DataFrame(predictions) 

output_dir = 'predictions'   
now = datetime.datetime.now()
timestamp_str = now.strftime("%m-%d_%H-%M")
filename = f"{pathology}_preds_{timestamp_str}.csv" 
os.makedirs(output_dir, exist_ok=True)
full_path = os.path.join(output_dir, filename) 
preds.to_csv(full_path, index=False) 