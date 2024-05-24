# Implementation of hierarchical clustering
# going to try and implement hierarchical modeling

print('Importing')
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.image as mpimg
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors

print('Done importing')

hpc = False
print(sys.argv)
if (len(sys.argv) > 1 and sys.argv[1] == 'hpc'):
    hpc = True


# In[1]:
lr = 0.0002
n_epochs = 10
batch_size = 256
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
print(hpc, device, n_epochs)


# In[2]:
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

transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class TrainImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, feature_extract_model=None):
        self.img_dir = img_dir
        self.transform = transform
        self.feature_extract_model = feature_extract_model
        if self.feature_extract_model:
            self.feature_extract_model.eval()

        if hpc:
            self.img_labels = parse_labels(pd.read_csv(annotations_file)[:-1])
        else:
            self.img_labels = parse_labels(pd.read_csv(annotations_file))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Path'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.feature_extract_model:
            with torch.no_grad():
                image = self.feature_extract_model(image.unsqueeze(0))

        label = torch.tensor(list(row[-9:])).float()  # extract label, the last 9 columns
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
        img_path = os.path.join(self.img_dir, row['Path'])
        img_id = row['Path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, img_id


# In[3]:

# In[4]
# Assuming ResNet18 is used for feature extraction
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()  # Modify the last layer to output the feature vector directly
resnet = resnet.to(device)

train_data = ImageDataset(labels_path_train, img_dir, transform, feature_extract_model=resnet)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = ImageDataset(labels_path_test, img_dir, transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# In[5]

train_features = []
train_ids = []
for images, ids in train_loader:
    images = images.to(device)
    outputs = resnet(images)
    train_features.extend(outputs.cpu().numpy())
    train_ids.extend(ids)

train_features = np.array(train_features)
linkage_matrix = linkage(train_features, method='ward')
train_cluster_labels = fcluster(linkage_matrix, t=15, criterion='distance')


features = []
ids = []
for images, image_ids in test_dataloader:
    images = images.to(device)
    with torch.no_grad():
        outputs = resnet(images)
    print("Batch output shape:", outputs.shape)
    if outputs.numel() == 0:
        print("Warning: No data in outputs.")
    features.extend(outputs.cpu().numpy())
    ids.extend(image_ids)

print("Collected features from", len(features), "images.")
"""
if features:
    features = [output.flatten() for output in features]
    features = np.vstack(features)
    print("Shape of features for clustering:", features.shape)

    if features.size > 0:
        linkage_matrix = linkage(features, method='ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()

"""

# In[6]

# make visualizations
features_array = np.array(features)
linkage_matrix = linkage(features_array, method='ward')

cutoff = 15  
cluster_labels = fcluster(linkage_matrix, cutoff, criterion='distance')

clustered_data = list(zip(ids, cluster_labels))
print(clustered_data)

def load_image(image_id, img_dir):
    img_path = os.path.join(img_dir, image_id)
    return mpimg.imread(img_path)

clustered_data.sort(key=lambda x: x[1])

cluster_sizes = defaultdict(int)
for _, cluster_label in clustered_data:
    cluster_sizes[cluster_label] += 1
max_images_in_cluster = max(cluster_sizes.values())

fig, axes = plt.subplots(nrows=max(cluster_sizes.keys()) + 1, ncols=max_images_in_cluster, figsize=(20, 10))
fig.suptitle('Images Grouped by Cluster', fontsize=16)

for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')

cluster_indices = defaultdict(int)
for image_id, cluster_label in clustered_data:
    row = cluster_label
    col = cluster_indices[cluster_label]
    ax = axes[row][col]
    img = load_image(image_id, img_dir)
    ax.imshow(img)
    ax.set_title(f'Cluster {cluster_label}', fontsize=10)
    ax.axis('on')  
    cluster_indices[cluster_label] += 1  

plt.tight_layout(pad=1.0)  
plt.show()

# In[7]

