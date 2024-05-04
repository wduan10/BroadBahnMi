#!/usr/bin/env python
# coding: utf-8

# In[3]:


print('Importing')
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
print('Done importing')


# In[39]:


pathology = 'Enlarged Cardiomediastinum'


# In[40]:


hpc = False
print(sys.argv)
if (len(sys.argv) > 1 and sys.argv[1] == 'hpc'):
    hpc = True


# In[41]:


lr = 0.0002
n_epochs = 1
batch_size = 256
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
print(hpc, device, n_epochs)


# In[42]:


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

df_train = df_train[['Path', pathology]]
df_train = df_train.dropna()
df_test = pd.read_csv(labels_path_test)
print(df_train.head())
print(df_test.head())


# In[43]:


def parse_labels(df):
    df = df[['Path', pathology]]
    df = df.dropna()
    return df

class TrainImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        if (hpc):
            self.img_labels = parse_labels(pd.read_csv(annotations_file)[:-1])
        else:
            self.img_labels = parse_labels(pd.read_csv(annotations_file))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]

        img_path = row['Path']
        img_path = os.path.join(self.img_dir, img_path)

        image = Image.open(img_path) # PIL image for applying transform for pre-trained ResNet model 
        label_num = row[-1] + 1 # -1 => 0, 0 => 1, 1 => 2
        label = torch.tensor(label_num).long()

        if self.transform:
            image = self.transform(image)

        return image, label
    
class TestImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]

        img_path = row['Path']
        img_path = os.path.join(self.img_dir, img_path)

        # image = read_image(img_path)
        image = Image.open(img_path) # PIL image for applying transform for pre-trained ResNet model 
        label = row['Id']

        if self.transform:
            image = self.transform(image)

        return image, label


# In[44]:


transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

training_data = TrainImageDataset(labels_path_train, img_dir, transform=transform)
test_data = TestImageDataset(labels_path_test, img_dir, transform=transform)

train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
training_data, val_data = torch.utils.data.random_split(training_data, [train_size, val_size])

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# In[45]:


# sample model with dropout = 0.1
model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=(3, 3)),
    nn.BatchNorm2d(num_features=8),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.1),

    nn.Conv2d(8, 4, kernel_size=(3, 3)),
    nn.BatchNorm2d(num_features=4),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.1),

    nn.Flatten(),
    nn.Linear(62*62*4, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
    # PyTorch implementation of cross-entropy loss includes softmax layer
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))


# In[46]:


# store metrics
training_loss_history = np.zeros(n_epochs)
validation_loss_history = np.zeros(n_epochs)

for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}/{n_epochs}:')
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


# In[48]:


# get predictions on test set
rows_list = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(test_dataloader):
        images, ids = data
        images, ids = images.to(device), ids.to(device)
        
        output = np.array(model(images).cpu())
        output = np.mean(output, axis=1)
        for preds, id in zip(output, ids):
            rows_list.append([int(id)] + [preds])

df_output = pd.DataFrame(rows_list, columns=['Id', pathology])
df_output.head()


# In[47]:


if (hpc):
    output_dir = '/groups/CS156b/2024/BroadBahnMi/predictions'
else:
    output_dir = 'predictions'

full_path = os.path.join(output_dir, f'preds_{pathology}.csv')
df_output.to_csv(full_path, index=False)


# In[ ]:




