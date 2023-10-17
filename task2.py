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

# Constants for dataset directory and classes
DATA_DIR = 'EuroSAT_RGB'
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Number of images to take from each class
num_images_per_class = 30

# Labelling Changes
LABEL_CHANGES = {
    'AnnualCrop': 'PermanentCrop',
    'PermanentCrop': 'AnnualCrop',
    'Forest': 'HerbaceousVegetation',
}

class EuroSATDataset(Dataset):
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

                # Modify the label based on the defined changes
                modified_label = self._modify_label(class_name)
                self.labels.append(modified_label)

    def __len__(self):
        return len(self.image_paths)
    
    def _modify_label(self, original_label):
        # Modify the label based on the defined changes
        if original_label in LABEL_CHANGES:
            original_label = LABEL_CHANGES[original_label]

        # Create a binary label vector
        modified_label = np.zeros(len(CLASSES), dtype=np.float32)
        modified_label[CLASSES.index(original_label)] = 1.0

        return modified_label
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)  # Use PIL to open the image
        image = image.convert('RGB')  # Convert to RGB if not already

        # Ensure 3 channels (RGB)
        image = np.asarray(image)
        if image.shape[2] != 3:
            # Convert grayscale images to RGB by duplicating the channels
            image = np.stack([image] * 3, axis=-1)
        
        # Resize the image to the desired dimensions
        image = cv2.resize(image, (64, 64))

        # Convert the image to a tensor
        image = Image.fromarray(image)  # Convert back to PIL Image
        if self.transform:
            image = self.transform(image)
        
        # Convert the image to a tensor
        image = transforms.ToTensor()(image)

        # Transpose to match the model's expected input shape
        image = image.permute(2, 0, 1)

        label = self.labels[idx]

        return image, label

def load_and_split_dataset(data_dir, random_seed=42):
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

def build_model(num_classes, input_channels=3, image_size=64):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Modify the fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()  # Use Sigmoid activation for multi-label classification
    )
    
    # Define the loss function and optimizer for multi-label classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.permute(0, 3, 1, 2)  # Change the dimension order
        images = images.float()  # Convert images to FloatTensor
        labels = labels.float()  # Convert labels to FloatTensor for multi-label
        images, labels = images.to(device), labels.to(device)  # Convert to appropriate device

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

def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.float()
            images = images.permute(0, 3, 1, 2)
            labels = labels.float()
            
            images, labels = images.to(device), labels.to(device)  # Convert to the appropriate device

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Collect predictions and true labels for metrics
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss for the epoch
    average_loss = running_loss / len(val_loader)

    # Convert predictions to binary labels (multi-label thresholding)
    binary_preds = np.array(all_preds) > 0.5

    # Convert all_labels and binary_preds to numpy arrays for correct indexing
    all_labels_np = np.array(all_labels)
    binary_preds_np = np.array(binary_preds)

    # Calculate average precision for each class
    avg_precision_per_class = []
    for class_idx in range(len(CLASSES)):
        avg_precision = average_precision_score(all_labels_np[:, class_idx], binary_preds_np[:, class_idx])
        avg_precision_per_class.append(avg_precision)

    # Compute accuracy per class
    accuracy_per_class = []
    for class_idx in range(len(CLASSES)):
        accuracy_class = accuracy_score(all_labels_np[:, class_idx], binary_preds_np[:, class_idx])
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
    dataset = EuroSATDataset(DATA_DIR, transform=transform)
    # print(f"Total number of samples in the dataset: {len(dataset)}")

    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_and_split_dataset(DATA_DIR)
    # Convert labels to binary format
    train_labels = label_binarize(train_labels, classes=list(range(len(CLASSES))))
    val_labels = label_binarize(val_labels, classes=list(range(len(CLASSES))))
    test_labels = label_binarize(test_labels, classes=list(range(len(CLASSES))))

    print(f"Number of samples in training set: {len(train_data)}")
    print(f"Number of samples in validation set: {len(val_data)}")
    print(f"Number of samples in test set: {len(test_data)}")

    model, criterion, optimizer = build_model(len(CLASSES), input_channels=3, image_size=64)
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

    augmentation_transforms = {
        'HorizontalFlip': transforms.Compose([transforms.RandomHorizontalFlip()]),
        'ColorJitter': transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]),
        'RandomRotation': transforms.Compose([transforms.RandomRotation(degrees=30)])
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
                EuroSATDataset(DATA_DIR, transform=augment_transform),
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

    # Plotting loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xticks(range(1, num_epochs+1)) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    