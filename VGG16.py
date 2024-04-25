import torch
import os 
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
hpc = (device == 'cuda')
BATCH_SIZE = 10
LEARNING_RATE = 0.0001 
NUM_EPOCHS = 5
N_CLASSES = 9 

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])  

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

        image = Image.open(img_path).convert('RGB')
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

        image = Image.open(img_path).convert('RGB')
        label = row['Id']

        if self.transform:
            image = self.transform(image)

        return image, label

classes = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

def train_model(model, optimizer, criterion, train_dataloader, val_dataloader, device):
    model.to(device)
    
    training_loss_history = np.zeros([NUM_EPOCHS, 1])
    validation_loss_history = np.zeros([NUM_EPOCHS, 1])

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/10:', end='')
        # train
        model.train()
        for i, data in enumerate(train_dataloader):
            images, labels = data
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            training_loss_history[epoch] += loss.item()

        training_loss_history[epoch] /= len(train_dataloader)
        print(f'Training Loss: {training_loss_history[epoch]}')

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
            print(f'Validation loss: {validation_loss_history[epoch]}')

    return model, training_loss_history, validation_loss_history 

training_data = TrainImageDataset(labels_path_train, img_dir, transform=transform)
test_data = TestImageDataset(labels_path_test, img_dir, transform=transform)

train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
training_data, val_data = torch.utils.data.random_split(training_data, [train_size, val_size])

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# model 
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, N_CLASSES),
)

optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

model, train_history, val_history = train_model(model, optimizer, criterion, train_dataloader, val_dataloader, device)

# get predictions on test set
rows_list = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(test_dataloader):
        images, ids = data
        images, ids = images.to(device), ids.to(device)
        
        output = model(images)
        for preds, id in zip(output, ids):
            preds = [float(x) for x in preds]
            rows_list.append([int(id)] + list(preds))

df_output = pd.DataFrame(rows_list, columns=['Id'] + list(df_train.columns[-9:]))
df_output.head()

if (hpc):
    output_dir = '/groups/CS156b/2024/BroadBahnMi/predictions'
else:
    output_dir = 'predictions'

number = 1
for file in os.listdir(output_dir):
    if (file[:5] == 'preds'):
        number = max(number, int(file[6:-4]) + 1)

full_path = os.path.join(output_dir, f'preds_{number}.csv')
df_output.to_csv(full_path, index=False)