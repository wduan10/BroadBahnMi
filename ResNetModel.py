import numpy as np
import sys 
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.utils
from torchvision import models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

lr = 0.0002
n_epochs = 5 
batch_size = 64
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
hpc = (device == 'cuda')

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

df_test = pd.read_csv(labels_path_test)
print(df_train.head())
print(df_test.head())

def parse_labels(df):
    df.fillna(0, inplace=True)
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

        image = Image.open(img_path) 
        label = torch.tensor(list(row[-9:])).float()  

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

        image = Image.open(img_path)  
        label = row['Id']

        if self.transform:
            image = self.transform(image)

        return image, label

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),  # pseudo-RGB
        transforms.ToTensor(),
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),   
        transforms.ToTensor(),
        normalize
    ])
}

train_dataset = TrainImageDataset(labels_path_train, img_dir, transform=data_transforms['train'])
val_dataset = TestImageDataset(labels_path_test, img_dir, transform=data_transforms['validation'])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
    'validation': DataLoader(val_dataset, batch_size=32, shuffle=False)
}

model = models.resnet50(pretrained=True)
model.to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 9)
)
model.fc.to(device) 

criterion = nn.MSELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0002)

for epoch in range(n_epochs):
    model.train()   
    running_loss = 0.0
    for images, labels in dataloaders['train']:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloaders['train'].dataset)
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}')