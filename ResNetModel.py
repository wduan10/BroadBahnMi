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

# Training local  
# labels_path_train = 'data/train/labels/labels.csv'
# img_dir_train = 'data/train/images'

# Testing local
# labels_path_test = 'data/test/ids.csv'
# img_dir_test = 'data/test/images'

# Training HPC
labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv'
img_dir_train = '/groups/CS156b/data/train'

# Testing HPC
labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv'
img_dir_test = '/groups/CS156b/data/test'

df_train = pd.read_csv(labels_path_train)[:-1]
df_test = pd.read_csv(labels_path_test)[:-1]

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)[:-1]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]

        img_path = row['Path'].split('/')
        img_path = '/'.join(img_path[1:])
        img_path = os.path.join(self.img_dir, img_path)

        image = read_image(img_path)
        # image = Image.open(img_path) # PIL image for applying transform for pre-trained ResNet model 
        image = Image.fromarray(np.stack((image,)*3, axis=-1)) # greyscale to RGB
        label = list(row[-9:]) # extract label, the last 9 columns

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# Transform for ResNet 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

training_data = CustomImageDataset(labels_path_train, img_dir_train, transform=transform)
test_data = CustomImageDataset(labels_path_test, img_dir_test, transform=transform)
a, b = training_data[0]
print('first item', a, b)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
for x, y in train_dataloader:
    print(y)
    break

classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

if torch.cuda.is_available():
    model.to('cuda')
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

n_epochs = 15 

for epoch in range(n_epochs):
    losses = []
    running_loss = 0
    for i, inp in enumerate(train_dataloader):
        inputs, labels = inp
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
    
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i%100 == 0 and i > 0:
            print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
            running_loss = 0.0

    avg_loss = sum(losses)/len(losses)
    scheduler.step(avg_loss)
            
print('Training Done')