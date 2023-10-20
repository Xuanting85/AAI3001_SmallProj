import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image 
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Constants for dataset directory and classes
DATA_DIR = 'EuroSAT_RGB'
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Number of images to take from each class
num_images_per_class = 50

class EuroSATDataset1(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load and preprocess the dataset
        self._load_dataset()

    def _load_dataset(self):
        for i, class_name in enumerate(CLASSES):
            class_path = os.path.join(self.data_dir, class_name)
            for image_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_name))
                self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)  # Use PIL to open the image
        image = image.convert('RGB')  # Convert to RGB if not already
        label = self.labels[idx]

        if self.transform:
            # Apply the transformation to get a tensor
            image = self.transform(image)

        # Convert the image to a tensor if not already
        image = transforms.ToTensor()(image)
    
        return image, label

class EuroSATDataset2(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load and preprocess the dataset
        self._load_dataset()

    def _load_dataset(self):
        for i, class_name in enumerate(CLASSES):
            class_path = os.path.join(self.data_dir, class_name)
            for image_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_name))
                self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)  # Use PIL to open the image
        image = image.convert('RGB')  # Convert to RGB if not already
        label = self.labels[idx]

        if self.transform:
            # Apply the transformation to get a tensor
            image = self.transform(image)

        # Convert the image to a tensor if not already
        image = transforms.ToTensor()(image)

        # One-hot encode the label
        label_one_hot = torch.eye(len(CLASSES))[int(label)]

        return image, label_one_hot

def load_and_split_dataset1(data_dir, random_seed=42):
    # Lists to store data and labels
    data = []
    labels = []

    # Load data and labels from each class
    for i, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        images_in_class = os.listdir(class_path)[:num_images_per_class]  # Take specified number of images per class

        for image_name in images_in_class:
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)[:, :, ::-1]  # Read image in BGR, convert to RGB
            image = cv2.resize(image, (64, 64))  # Resize image
            data.append(image)
            labels.append(i)

    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split into train, validation, and test sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.3, random_state=random_seed)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=random_seed)

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)



def load_and_split_dataset2(data_dir, random_seed=42):
    # Lists to store data and modified labels
    data = []
    modified_labels = []

    # Load data and labels from each class
    for i, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        images_in_class = os.listdir(class_path)[:num_images_per_class]  # Take specified number of images per class

        for image_name in images_in_class:
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)[:, :, ::-1]  # Read image in BGR, convert to RGB
            image = cv2.resize(image, (64, 64))  # Resize image
            data.append(image)

            # Apply label modifications
            modified_label = [0] * len(CLASSES)
            modified_label[i] = 1

            if class_name == 'AnnualCrop':
                modified_label[6] = 1  # Set PermanentCrop to 1
            elif class_name == 'PermanentCrop':
                modified_label[0] = 1  # Set AnnualCrop to 1
            elif class_name == 'Forest':
                modified_label[2] = 1  # Set HerbaceousVegetation to 1

            modified_labels.append(modified_label)

    # Convert data and labels to numpy arrays
    data = np.array(data)
    modified_labels = np.array(modified_labels)

    # Split into train, validation, and test sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, modified_labels, test_size=0.3, random_state=random_seed)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=random_seed)

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
