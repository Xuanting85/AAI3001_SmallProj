import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score


# Set the random seed for reproducibility
random_seed = 42

class EuroSATDataset(Dataset):
    def __init__(self, data_dir, transform=None, random_seed=42, num_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        self.random_seed = random_seed
        self.classes = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", 
            "Highway", "Industrial", "Pasture", 
            "PermanentCrop", "Residential", "River", "SeaLake"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images, self.labels = self.load_data(num_samples=num_samples)

    def load_data(self, num_samples=None):
        images = []
        labels = []
        for idx, cls in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, cls)
            for img_name in os.listdir(class_path)[:num_samples]:
                img_path = os.path.join(class_path, img_name)
                images.append(img_path)
                labels.append(idx)
        return images, labels

    def split_dataset(self, random_seed=42):
        # Split the dataset into train, validation, and test sets
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            self.images, self.labels, test_size=0.2, random_state=self.random_seed)
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=0.5, random_state=self.random_seed)

        return train_images, val_images, test_images, train_labels, val_labels, test_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    train_losses = []  # to store training loss per epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return train_losses

def validate_model(model, val_loader, criterion, device):
    model.to(device)
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_loss = val_loss / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Calculate average precision per class
    average_precision_per_class = calculate_average_precision_per_class(all_labels, outputs)

    # Calculate accuracy per class
    accuracy_per_class = calculate_accuracy_per_class(all_labels, all_preds)

    # Compute the average accuracy over all classes
    average_accuracy = sum(accuracy_per_class.values()) / len(accuracy_per_class)

    # Store these metrics for analysis
    metrics = {
        'average_precision_per_class': average_precision_per_class,
        'accuracy_per_class': accuracy_per_class,
        'average_accuracy': average_accuracy
    }

    return val_loss, val_accuracy, metrics

def calculate_average_precision_per_class(labels, outputs):
    # Calculate average precision per class
    num_classes = outputs.shape[1]
    average_precision_per_class = {}
    for class_idx in range(num_classes):
        binary_labels = [1 if label == class_idx else 0 for label in labels]
        binary_outputs = outputs[:, class_idx]
        average_precision = average_precision_score(binary_labels, binary_outputs)
        average_precision_per_class[class_idx] = average_precision

    return average_precision_per_class

def calculate_accuracy_per_class(labels, predictions):
    # Calculate accuracy per class
    num_classes = len(set(labels))
    accuracy_per_class = {}
    for class_idx in range(num_classes):
        true_positive = sum((labels == class_idx) & (predictions == class_idx))
        total_true = sum(labels == class_idx)
        accuracy_per_class[class_idx] = true_positive / total_true if total_true != 0 else 0

    return accuracy_per_class

def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")


def main():
    print("EuroSAT Dataset Classification")
    print("------------------------------")

    # Load and preprocess a small subset of the dataset for quick testing
    num_samples = 10  # Adjust this number as needed
    data_dir = 'EuroSAT_RGB'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eurosat_dataset = EuroSATDataset(data_dir, transform=transform, random_seed=random_seed, num_samples=num_samples)
    train_images, val_images, test_images, train_labels, val_labels, test_labels = eurosat_dataset.split_dataset()

    # Create the model
    print("Creating the model...")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_classes = len(eurosat_dataset.classes)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Training parameters
    learning_rate = 0.001
    batch_size = 4
    num_epochs = 3
    device = torch.device("cpu")

    # Data loaders
    train_loader = DataLoader(eurosat_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(eurosat_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(eurosat_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model and get training losses
    print("Training the model...")
    train_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    # Validate the model
    print("Validating the model...")
    val_loss, val_accuracy, metrics = validate_model(model, val_loader, criterion, device)

    # Test the model
    print("Testing the model...")
    test_model(model, test_loader, device)

    # Print training losses
    print("Training Losses:")
    print(train_losses)

    # Print validation metrics
    print("Validation Loss:", val_loss)
    print("Validation Accuracy:", val_accuracy)
    print("Average Precision per Class:", metrics['average_precision_per_class'])
    print("Accuracy per Class:", metrics['accuracy_per_class'])
    print("Average Accuracy over all Classes:", metrics['average_accuracy'])


if __name__ == "__main__":
    main()
