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
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
# from keras.src.applications.densenet import DenseNet121, preprocess_input 
# from keras.src.applications.resnet import ResNet18, preprocess_input
# from keras.src.applications.vgg16 import VGG16, preprocess_input
from keras.src.applications.inception_v3 import InceptionV3

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

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       zoom_range=0.1,
                                       horizontal_flip=True)

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

# Fine tuning InceptionV3 
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Adding a fully-connected layer
x = Dense(1024, activation='relu')(x)
# Adding a logistic layer  
predictions = Dense(3, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(train_data,
    epochs=NUM_EPOCHS,
    validation_data=val_data,
    verbose=1)

# This will show you the mapping between the one-hot encoded vectors and your original labels.
print(train_data.class_indices)

predictions = model.predict(test_data)
preds = pd.DataFrame(predictions) 
output_dir = 'predictions'   
now = datetime.datetime.now()
timestamp_str = now.strftime("%m-%d_%H-%M")
filename = f"{pathology}_preds_{timestamp_str}.csv" 
os.makedirs(output_dir, exist_ok=True)
full_path = os.path.join(output_dir, filename) 
preds.to_csv(full_path, index=False) 