import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# Set the random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Paths to dataset
root_dir = "EuroSAT_RGB"

# Load the EuroSAT dataset and split into train, validation, and test sets
def load_and_split_data(root_dir, random_seed):
    # Load dataset and obtain class labels
    dataset = ImageFolder(root_dir)
    labels = dataset.classes

    # Split the dataset into train, validation, and test sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        dataset.imgs, dataset.targets, test_size=0.2, random_state=random_seed, stratify=dataset.targets)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=random_seed, stratify=temp_labels)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, labels

# Custom dataset class
class EuroSATdataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index], self.labels[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Binarize labels using LabelBinarizer
def binarize_labels(labels):
    binarizer = LabelBinarizer()
    binarized_labels = binarizer.fit_transform(labels)
    return binarized_labels

# Data augmentation and transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and split the data
train_data, train_labels, val_data, val_labels, test_data, test_labels, class_labels = load_and_split_data(root_dir, random_seed)

# Binarize labels
train_labels_bin = binarize_labels(train_labels)
val_labels_bin = binarize_labels(val_labels)
test_labels_bin = binarize_labels(test_labels)

# Create custom datasets
train_dataset = EuroSATdataset(train_data, train_labels_bin, transform)
val_dataset = EuroSATdataset(val_data, val_labels_bin, transform)
test_dataset = EuroSATdataset(test_data, test_labels_bin, transform)

# Print the number of images in each dataset
print(f"Number of images in the train dataset: {len(train_dataset)}")
print(f"Number of images in the validation dataset: {len(val_dataset)}")
print(f"Number of images in the test dataset: {len(test_dataset)}")

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load pretrained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the last fully connected layer for 10 classes
model.fc = nn.Linear(model.fc.in_features, len(class_labels))

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print the modified model architecture
# print(model)

# Rest of the code would include training, validation, and testing loops, as well as evaluation and visualization.
