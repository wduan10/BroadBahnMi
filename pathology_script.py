import os 
import datetime 
import pandas as pd
from IPython.display import display
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras 
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import Model
from keras import layers 
from sklearn.model_selection import train_test_split
from keras.src.applications.vgg16 import VGG16, preprocess_input

BATCH_SIZE = 256 
NUM_EPOCHS = 10  
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
    # pathology = 'Fracture'
    df['filename'] = df_train['Path']
    df['label'] = df_train[pathology]

    if (gpus):
        df['label'] = parse_labels(df['label'][:-1])
    else:
        df['label'] = parse_labels(df['label'])

    # 'categorical' requires strings
    df['label'] = df['label'].astype(str)

    # Stratified train/test split based on 'Frontal/Lateral' column
    train_df, val_df = train_test_split(df,
                                        test_size=TEST_SIZE,
                                        random_state=42, 
                                        stratify=df['label'])

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rescale=1./255, #Normalize
                                    zoom_range=0.4,
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

pathology = "Pleural Other"
train_data, val_data = get_pathology(pathology)

# VGG16 Model
conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# Customize top layer
top_layer = conv_base.output
top_layer = keras.layers.GlobalAveragePooling2D()(top_layer)
top_layer = keras.layers.Dense(4096, activation='relu')(top_layer)
top_layer = keras.layers.Dense(1072, activation='relu')(top_layer)
top_layer = keras.layers.Dropout(0.2)(top_layer)
output_layer = keras.layers.Dense(3, activation='softmax')(top_layer) # Predicting for one pathology 

model = Model(inputs=conv_base.input, outputs=output_layer)

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# For lower num_epochs 
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=LEARNING_RATE,
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_data,
    epochs=NUM_EPOCHS,
    validation_data=val_data,
    verbose=1)

model.evaluate(val_data)

# Dataframe of predictions for 3 classes 
predictions = model.predict(test_data)
columns = list(val_data.class_indices.keys()) 
preds = pd.DataFrame(predictions, columns=columns)

output_dir = 'predictions'   
now = datetime.datetime.now()
timestamp_str = now.strftime("%m-%d_%H-%M")
filename = f"{pathology}_preds_{timestamp_str}.csv" 
os.makedirs(output_dir, exist_ok=True)
full_path = os.path.join(output_dir, filename) 
preds.to_csv(full_path, index=False) 