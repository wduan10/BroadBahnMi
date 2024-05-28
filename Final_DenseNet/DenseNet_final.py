#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
from DenseNet_model import DenseNet
from PIL import Image
from datetime import datetime 
print('Done importing')


# In[18]:


start_time = datetime.now()
print(start_time)


# In[19]:


pathologies = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
               'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other',
               'Fracture', 'Support Devices']
hpc = False
mode = 1
print(sys.argv)
if (len(sys.argv) > 1 and sys.argv[1] == 'hpc'):
    hpc = True
    if (len(sys.argv) > 2):
        mode = int(sys.argv[2])

pathology = pathologies[mode]
print('pathology:', pathology)


# In[20]:


lr = 0.0002
n_epochs = 30
n_cpu = 4 if hpc else 0
batch_size = 128
img_size = 256
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
print(hpc, device, n_epochs, n_cpu, img_size)


# In[21]:


if (hpc):
    labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv'
    labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv'
    img_dir = '/groups/CS156b/data'

    df_train = pd.read_csv(labels_path_train)[:-1]
else:
    labels_path_train = '../data/train/labels/labels.csv'
    labels_path_test = '../data/test/ids.csv'
    img_dir = '../data'

    df_train = pd.read_csv(labels_path_train)

df_train = df_train[['Path', pathology]]
df_train = df_train.dropna()
df_test = pd.read_csv(labels_path_test)
print(df_train.head())
print(df_test.head())


# In[22]:


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


# In[23]:


# transform with random flipping and cropping:
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((256, 256)),
    # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
])

training_data = TrainImageDataset(labels_path_train, img_dir, transform=transform)
test_data = TestImageDataset(labels_path_test, img_dir, transform=transform)

train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
training_data, val_data = torch.utils.data.random_split(training_data, [train_size, val_size])

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# In[24]:


model = DenseNet(channels=3, growth_rate=16, num_classes=3)
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))


# In[27]:


# store metrics
training_loss_history = np.zeros(n_epochs)
validation_loss_history = np.zeros(n_epochs)
early_stop = False

for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}/{n_epochs}:')
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

        # check if stop
        current_time = datetime.now()
        time_difference = current_time - start_time
        duration_in_s = time_difference.total_seconds() 
        hours = divmod(duration_in_s, 3600)[0]
        if (hours > 12):
            early_stop = True
            break
    
    training_loss_history[epoch] /= len(train_dataloader)
    print(f'Training Loss: {training_loss_history[epoch]:0.4f}')

    if (early_stop):
        break

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


# In[13]:


# get predictions on test set
rows_list = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(test_dataloader):
        images, ids = data
        images, ids = images.to(device), ids.to(device)
        
        output = model(images).cpu()
        output = nn.functional.softmax(output)
        output = (-1 * output[:, 0] + output[:, 2]) / 2
        for preds, id in zip(output, ids):
            rows_list.append([int(id)] + [float(preds)])

df_output = pd.DataFrame(rows_list, columns=['Id', pathology])
df_output.head()


# In[29]:


if (hpc):
    output_dir = '/groups/CS156b/2024/BroadBahnMi/predictions'
else:
    output_dir = '../predictions'

time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

filename = '_'.join(pathology.split()) + '_' + time
filename = filename.replace(' ', '_')
full_path = os.path.join(output_dir, f'preds_{filename}.csv')
df_output.to_csv(full_path, index=False)


# In[ ]:




