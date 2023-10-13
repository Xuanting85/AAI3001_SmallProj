import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from torchvision import transforms, datasets, models
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, accuracy_score

# Constants for dataset directories and classes
DATA_DIR = 'EuroSAT_RGB'
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Task 1: Custom Dataset class implementation
class EuroSATDataset(torch.utils.data.Dataset):
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
        image = cv2.imread(image_path)[:,:,::-1]  # Read image in BGR, convert to RGB
        image = cv2.resize(image, (224, 224))  # Resize image to desired size
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Task 2: Data Preparation and Splitting
def load_and_split_dataset(data_dir, val_split=0.15, test_split=0.15, random_seed=42, small_portion=1.0):
    # Load the EuroSAT dataset and split into train, validation, and test sets
    dataset = EuroSATDataset(data_dir)

    # If using a small portion of the dataset
    if small_portion < 1.0:
        total_samples = len(dataset)
        total_samples_to_use = int(total_samples * small_portion)
        dataset, _ = train_test_split(dataset, train_size=total_samples_to_use, random_state=random_seed)

    # Obtain labels and data
    labels = np.array([label for _, label in dataset])
    data = [data for data, _ in dataset]

    # Calculate the number of samples for each split
    total_samples = len(dataset)
    train_samples = int(total_samples * (1 - val_split - test_split))
    val_samples = int(total_samples * val_split)
    test_samples = total_samples - train_samples - val_samples

    # Split into train, validation, and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_samples, random_state=random_seed)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_samples, random_state=random_seed)

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


# Task 3: Model Architecture and Training
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


# Task 4: Training Loop
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


# Task 5: Validation Loop
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


# Main function to orchestrate the project
def main():
    print("EuroSAT Dataset Classification")
    print("=" * 50)

    # Task 1: Custom Dataset class implementation
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = EuroSATDataset(DATA_DIR, transform=transform)
    # print(f"Total number of samples in the dataset: {len(dataset)}")

    # Task 2: Data Preparation and Splitting
    small_portion = 0.1  # Change this to the desired portion (e.g., 0.1 for 10% of the dataset)
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_and_split_dataset(DATA_DIR, small_portion=small_portion)
    print(f"Number of samples in training set: {len(train_data)}")
    print(f"Number of samples in validation set: {len(val_data)}")
    print(f"Number of samples in test set: {len(test_data)}")

    # Task 3: Model Architecture and Training
    model, criterion, optimizer = build_model(len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\n\nTraining the model...")

    # Convert data to numpy arrays before creating tensors
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # Create DataLoader for training set
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create DataLoader for validation set
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_data), torch.tensor(val_labels))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # Create DataLoader for test set
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Lists to store loss for plotting
    train_losses = []
    val_losses = []

    # Task 4: Training Loop
    num_epochs = 3  # Adjust the number of epochs as needed
    for epoch in range(num_epochs):
        print("=" * 50)

        # Train the model for one epoch
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        # Task 5: Validation Loop
        val_loss, val_avg_precision, val_accuracy, avg_accuracy = validate_model(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Average Precision per Class: {val_avg_precision}")
        print(f"Validation Accuracy per Class: {val_accuracy}")
        print(f"Average Validation Accuracy: {avg_accuracy:.4f}")
        val_losses.append(val_loss)

    # Plotting loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Task 7: Testing
    # Test the final model on the test data
    test_loss, test_avg_precision, test_accuracy, avg_test_accuracy = validate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Average Precision per Class: {test_avg_precision}")
    print(f"Test Accuracy per Class: {test_accuracy}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")


if __name__ == "__main__":
    main()
