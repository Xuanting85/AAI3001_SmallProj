import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, accuracy_score

import utils
from utils import load_and_split_dataset1, EuroSATDataset1,DATA_DIR, CLASSES

def build_model(num_classes):
    # Load the pre-trained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze all layers except the final classification layer
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the final classification layer for the specified number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    return model, criterion, optimizer

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.permute(0, 3, 1, 2)  # Change the dimension order
        images = images.float()  # Convert images to FloatTensor
        images = images[:, :3, :, :]  # Keep only the first 3 channels (for RGB)
        images, labels = images.to(device), labels.long().to(device)  # Convert labels to LongTensor

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss for reporting
        running_loss += loss.item()

    # Calculate average loss for the epoch
    average_loss = running_loss / len(train_loader)

    return average_loss


def main():
    print("EuroSAT Dataset Classification")
    print("=" * 50)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = EuroSATDataset1(DATA_DIR, transform=transform)
    # print(f"Total number of samples in the dataset: {len(dataset)}")

    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_and_split_dataset1(DATA_DIR)
    print(f"Number of samples in training set: {len(train_data)}")

    model, criterion, optimizer = build_model(len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\nTraining the model...")

    # Convert data to numpy arrays before creating tensors
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # Create DataLoader for training set
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Lists to store loss for plotting
    train_losses = []

    num_epochs = 3

    val_transform_set1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    ])

    val_transform_set2 = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.CenterCrop(180),
    ])

    val_transform_set3 = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomInvert(p=0.5),
        transforms.Resize(224),
    ])

    # Create a dictionary to associate transformation names with their corresponding transformations
    augmentation_transforms = {
        'Augmentation Set 1': val_transform_set1,
        'Augmentation Set 2': val_transform_set2,
        'Augmentation Set 3': val_transform_set3,
    }

    for epoch in range(num_epochs):
        print("=" * 50)

        # Train the model for one epoch
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        # Plotting loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.xticks(range(1, num_epochs+1)) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.show()
    
if __name__ == "__main__":
    main()
    