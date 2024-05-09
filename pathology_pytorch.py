import torch
import os
import datetime 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd 
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

classes = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
           "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

NUM_EPOCHS = 1 
BATCH_SIZE = 64 
LEARNING_RATE = 0.0002
HPC = True 
IMAGE_SIZE = 224
NUM_CLASSES = 3
PATHOLOGY = "Fracture"

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
print(f"Device: {device}")

if (HPC):
    labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv'
    labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv'
    img_dir = '/groups/CS156b/data'

    df_train = pd.read_csv(labels_path_train)[:-1]    

    TRAIN_SIZE = 0.8
else:
    labels_path_train = '/content/BroadBahnMi/data/train/labels/labels.csv'
    labels_path_test = '/content/BroadBahnMi/data/test/ids.csv'
    img_dir = '/content/BroadBahnMi/data'

    df_train = pd.read_csv(labels_path_train)
    TRAIN_SIZE = 0.5 

df_test = pd.read_csv(labels_path_test)
df_test_pathology = df_test[['Path', 'Id']]

df_train_pathology = df_train[['Path', PATHOLOGY]] 

# replace nan with zeros 
df_train_pathology[PATHOLOGY].fillna(0, inplace=True) 

class PathologyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, test=False):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['Path'])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        if not self.test:
            label = self.df.iloc[idx][1] + 1 # label out of bounds 
            label = torch.tensor(label).long()
            return image, label
        else:
            # Return the image and ID for test samples
            id = self.df.iloc[idx]['Id']
            return image, id
        
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Normalize((0.5,), (0.5,))
        ])

# Dataloader
train_data = PathologyDataset(df_train_pathology, img_dir, transform=transform)
test_data = PathologyDataset(df_test_pathology, img_dir, transform=transform)

total_size = len(train_data)
train_size = int(total_size * TRAIN_SIZE)
test_size = total_size - train_size

train_dataset, val_dataset = random_split(train_data, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Model 
# Transfer learning fine tuning approach 
model = models.densenet121(pretrained=True)
# Replace first convolutional layer to accept greyscale images
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

for param in model.parameters():
    # Freezes weights 
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, NUM_CLASSES),
                                 )

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

training_loss_history = np.zeros(NUM_EPOCHS)
validation_loss_history = np.zeros(NUM_EPOCHS)

# From CNN_single.py
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
    train_total = 0
    train_correct = 0
    # train
    model.train()
    for i, data in enumerate(train_dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward pass
        output = model(images)
        # calculate categorical cross entropy loss
        loss = criterion(output, labels)
        # backward pass
        loss.backward()
        optimizer.step()

        # track training loss
        training_loss_history[epoch] += loss.item()
    
    training_loss_history[epoch] /= len(train_dataloader)
    print(f'Training Loss: {training_loss_history[epoch]:0.4f}')

    # validate
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(val_dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # forward pass
            output = model(images)
            # find loss
            loss = criterion(output, labels)
            validation_loss_history[epoch] += loss.item()
        validation_loss_history[epoch] /= len(val_dataloader)
    print(f'Validation loss: {validation_loss_history[epoch]:0.4f}')

# Predictions on the test set 
rows_list = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(test_dataloader):
        images, ids = data
        images, ids = images.to(device), ids.to(device)
        
        output = np.array(model(images).cpu())
        output = np.argmax(output, axis=1) - 1
        for preds, id in zip(output, ids):
            rows_list.append([int(id)] + [preds])

df_output = pd.DataFrame(rows_list, columns=['Id', PATHOLOGY])
df_output.head()

output_dir = 'predictions'   
now = datetime.datetime.now()
timestamp_str = now.strftime("%m-%d_%H-%M")
filename = f"{PATHOLOGY}_preds_{timestamp_str}.csv" 
os.makedirs(output_dir, exist_ok=True)
full_path = os.path.join(output_dir, filename) 
df_output.to_csv(full_path, index=False) 