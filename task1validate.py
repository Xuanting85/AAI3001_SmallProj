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
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, accuracy_score

import utils
from task1train import build_model,train_model
from utils import load_and_split_dataset1, EuroSATDataset1, DATA_DIR, CLASSES

def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.float()
            images = images.permute(0, 3, 1, 2)
            images = images[:, :3, :, :]
            images, labels = images.to(device), labels.long().to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Collect predictions and true labels for metrics
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss for the epoch
    average_loss = running_loss / len(val_loader)

    # Convert labels and predictions to one-hot encoding for average_precision_score
    one_hot_labels = label_binarize(all_labels, classes=list(range(len(CLASSES))))
    one_hot_preds = label_binarize(all_preds, classes=list(range(len(CLASSES))))

    # Calculate average precision for each class
    avg_precision_per_class = []
    for class_idx in range(len(CLASSES)):
        avg_precision = average_precision_score(one_hot_labels[:, class_idx], one_hot_preds[:, class_idx])
        avg_precision_per_class.append(avg_precision)

    # Compute accuracy per class
    accuracy_per_class = []
    for class_idx in range(len(CLASSES)):
        accuracy_class = accuracy_score(one_hot_labels[:, class_idx], one_hot_preds[:, class_idx])
        accuracy_per_class.append(accuracy_class)

    # Compute the average accuracy over all classes
    avg_accuracy = sum(accuracy_per_class) / len(CLASSES)

    return average_loss, avg_precision_per_class, accuracy_per_class, avg_accuracy


def test_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float()
            images = images.permute(0, 3, 1, 2)
            images = images[:, :3, :, :]
            images, labels = images.to(device), labels.long().to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Collect predictions and true labels for metrics
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss for the epoch
    average_loss = running_loss / len(test_loader)

    # Convert labels and predictions to one-hot encoding for average_precision_score
    one_hot_labels = label_binarize(all_labels, classes=list(range(len(CLASSES))))
    one_hot_preds = label_binarize(all_preds, classes=list(range(len(CLASSES))))

    # Calculate average precision for each class
    avg_precision_per_class = []
    for class_idx in range(len(CLASSES)):
        avg_precision = average_precision_score(one_hot_labels[:, class_idx], one_hot_preds[:, class_idx])
        avg_precision_per_class.append(avg_precision)

    # Compute accuracy per class
    accuracy_per_class = []
    for class_idx in range(len(CLASSES)):
        accuracy_class = accuracy_score(one_hot_labels[:, class_idx], one_hot_preds[:, class_idx])
        accuracy_per_class.append(accuracy_class)

    # Compute the average accuracy over all classes
    avg_accuracy = sum(accuracy_per_class) / len(CLASSES)

    return average_loss, avg_precision_per_class, accuracy_per_class, avg_accuracy


def main():
    print("EuroSAT Dataset Classification")
    print("=" * 50)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = EuroSATDataset1(DATA_DIR, transform=transform)
    # print(f"Total number of samples in the dataset: {len(dataset)}")

    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_and_split_dataset1(DATA_DIR)
    print(f"Number of samples in training set: {len(train_data)}")
    print(f"Number of samples in validation set: {len(val_data)}")
    print(f"Number of samples in test set: {len(test_data)}")

    model, criterion, optimizer = build_model(len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\nTraining the model...")

    # Convert data to numpy arrays before creating tensors
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # Create DataLoader for training set
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create DataLoader for validation set
    val_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create DataLoader for test set
    test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Lists to store loss for plotting
    train_losses = []
    val_losses = []

    num_epochs = 3
    best_val_loss = float('inf')

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

        # Validation Loop with data augmentation
        val_loss, val_avg_precision, val_accuracy, avg_accuracy = validate_model(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Average Precision per Class: {val_avg_precision}")
        print(f"Validation Accuracy per Class: {val_accuracy}")
        print(f"Average Validation Accuracy: {avg_accuracy:.4f}")

        # Apply data augmentation to the validation dataset for each type
        for augment_name, augment_transform in augmentation_transforms.items():
            augmented_val_loader = DataLoader(
                EuroSATDataset1(DATA_DIR, transform=augment_transform),
                batch_size=32
            )
            augmented_val_loss, aug_val_avg_precision, aug_val_accuracy, avg_aug_val_accuracy = validate_model(model, augmented_val_loader, criterion, device)
    
            print(f"\nValidation with {augment_name}")
            print(f"Validation Loss: {augmented_val_loss:.4f}")
            print(f"Validation Average Precision per Class: {aug_val_avg_precision}")
            print(f"Validation Accuracy per Class: {aug_val_accuracy}")
            print(f"Average Validation Accuracy: {avg_aug_val_accuracy:.4f}")

        val_losses.append(val_loss)

        # Check if this is the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    
    print("=" * 50)

    # Test the best model on the test set
    test_loss, test_avg_precision, test_accuracy, avg_test_accuracy = test_model(best_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Average Precision per Class: {test_avg_precision}")
    print(f"Test Accuracy per Class: {test_accuracy}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")

    # Save the trained model
    model_name = 'image_resnet50_model.pth'
    torch.save(best_model.state_dict(), model_name)
    print(f"Best model saved as {model_name}")

    # Plotting loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xticks(range(1, num_epochs+1)) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    

    plot_filename = 'Task 1_Training and Validation Loss Curves.png'
    plt.savefig(plot_filename)
    print(f"Loss curve plot saved as {plot_filename}")

    plt.show()


if __name__ == "__main__":
    main()
    