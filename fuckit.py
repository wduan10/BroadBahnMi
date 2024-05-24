import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
print('Importing libraries... Done.')

# Setup check
hpc = False
if len(sys.argv) > 1 and sys.argv[1] == 'hpc':
    hpc = True

# Data paths
labels_path_train = '/groups/CS156b/data/student_labels/train2023.csv' if hpc else 'data/train/labels/labels.csv'
labels_path_test = '/groups/CS156b/data/student_labels/test_ids.csv' if hpc else 'data/test/ids.csv'
img_dir = '/groups/CS156b/data' if hpc else 'data'

# Load data
df_train = pd.read_csv(labels_path_train)
df_test = pd.read_csv(labels_path_test)

# Data preprocessing
def parse_labels(df):
    df.fillna(0, inplace=True)
    return df  # Adjust this slice according to your actual data structure

class TrainImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform):
        if hpc:
            self.image_labels = parse_labels(pd.read_csv(annotations_file)[:-1])
        else:
            self.img_labels = parse_labels(pd.read_csv(annotations_file))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        image_path = os.path.join(self.img_dir, row['Path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        labels = torch.tensor(row[-9:].values.astype(np.float32))
        return image, labels

class TestImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        image_path = os.path.join(self.img_dir, row['Path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, row['Id']

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Feature extraction setup
resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)
resnet.eval()

# Extract features and pathologies
def extract_features_and_pathologies(loader):
    features = []
    pathologies = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)  # Correct usage: Use 'device' directly
            feats = resnet(images)  # Process images through the network
            features.append(feats.cpu().numpy())  # Convert features to NumPy arrays after moving them back to CPU
            pathologies.append(labels.cpu().numpy())  # Same for labels
    return np.vstack(features), np.vstack(pathologies)
    
# Load data and prepare DataLoader
train_dataset = TrainImageDataset(labels_path_train, img_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_features, train_pathologies = extract_features_and_pathologies(train_loader)

# Hierarchical clustering
linkage_matrix = linkage(train_features, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Assign clusters
cluster_labels = fcluster(linkage_matrix, t=5, criterion='distance')

# Calculate mean pathologies for each cluster
cluster_pathologies = {}
for idx, cluster in enumerate(cluster_labels):
    if cluster not in cluster_pathologies:
        cluster_pathologies[cluster] = []
    cluster_pathologies[cluster].append(train_pathologies[idx])
cluster_means = {cluster: np.mean(vals, axis=0) for cluster, vals in cluster_pathologies.items()}

# Classify test data
test_dataset = TestImageDataset(labels_path_test, img_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_features, test_ids = extract_features_and_pathologies(test_loader)

# Nearest neighbors for cluster assignment
nn_model = NearestNeighbors(n_neighbors=1)
nn_model.fit(train_features)
_, indices = nn_model.kneighbors(test_features)
predicted_pathologies = [cluster_means[cluster_labels[idx]] for idx in indices.flatten()]

# Output results to CSV
df_results = pd.DataFrame(predicted_pathologies, columns=df_train.columns[-9:])
test_ids = test_ids.flatten()
df_results.insert(0, 'Id', test_ids)


#df_results.to_csv('predicted_pathologies_cluster.csv', index=False)
#print("CSV file has been created.")



if (hpc):
    output_dir = '/groups/CS156b/2024/BroadBahnMi/predictions'
else:
    output_dir = 'predictions'

filename = 'pathologies_cluster.csv'
full_path = os.path.join(output_dir, f'preds_{filename}.csv')
df_results.to_csv(full_path, index=False)