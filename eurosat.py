import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class EuroSATDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", 
            "Highway", "Industrial", "Pasture", 
            "PermanentCrop", "Residential", "River", "SeaLake"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images, self.labels = self.load_data()

    def load_data(self):
        images = []
        labels = []
        for idx, cls in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append(img_path)
                labels.append(idx)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label