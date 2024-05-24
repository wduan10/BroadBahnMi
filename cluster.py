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
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Path'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = row['Label']  # Assuming 'Label' is the column with pathology labels
        return image, label
# In[3]:

# In[4]
# Assuming ResNet18 is used for feature extraction
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()  # Modify the last layer to output the feature vector directly
resnet = resnet.to(device)

train_data = ImageDataset(labels_path_train, img_dir, transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = ImageDataset(labels_path_test, img_dir, transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# In[5]
train_features = []
train_ids = []
resnet.eval()
with torch.no_grad():
    for images, ids in train_loader:
        images = images.to(device)
        features = resnet(images)
        train_features.extend(features.cpu().numpy())
        train_ids.extend(ids.numpy())

train_features = np.array(train_features)
linkage_matrix = linkage(train_features, method='ward')
train_cluster_labels = fcluster(linkage_matrix, t=15, criterion='distance')

# Assign most common pathology to each cluster
cluster_to_pathology = defaultdict(list)
for label, cluster in zip(train_ids, train_cluster_labels):
    cluster_to_pathology[cluster].append(label)
cluster_to_common_pathology = {cluster: Counter(pathologies).most_common(1)[0][0] for cluster, pathologies in cluster_to_pathology.items()}
# Classify test images
nbrs = NearestNeighbors(n_neighbors=1).fit(train_features)
test_features = []
test_ids = []
with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        features = resnet(images)
        test_features.extend(features.cpu().numpy())
        test_ids.extend(ids)

distances, indices = nbrs.kneighbors(test_features)
predicted_clusters = train_cluster_labels[indices.flatten()]
predicted_pathologies = [cluster_to_common_pathology[cluster] for cluster in predicted_clusters]

# Output results
for img_id, pathology in zip(test_ids, predicted_pathologies):
    print(f"Image ID: {img_id}, Predicted Pathology: {pathology}")

# Optionally, visualize results
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(range(len(predicted_pathologies)), predicted_pathologies, c=predicted_clusters)
plt.show()