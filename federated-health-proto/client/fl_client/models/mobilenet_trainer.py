"""
MobileNetV2 trainer for blood cell image classification (Diagnostic Labs).
Uses frozen backbone + trainable head for CPU efficiency.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List
import io
import os


class ImageDataset(Dataset):
    """Simple image dataset"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MobileNetV2Trainer:
    """Trainer for image classification using MobileNetV2"""
    
    def __init__(
        self,
        num_classes: int = 8,  # BCCD has 8 classes
        image_size: int = 96,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        num_epochs: int = 3,
        num_threads: int = 1,
        freeze_backbone: bool = True
    ):
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set CPU threads
        torch.set_num_threads(num_threads)
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Model
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
        
        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.model.eval()
    
    def train(
        self,
        image_paths: List[str],
        labels: List[int]
    ) -> Dict[str, float]:
        """Train model on local images"""
        
        # Create dataset
        dataset = ImageDataset(image_paths, labels, self.transform)
        
        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )
        
        # Train
        self.model.train()
        for epoch in range(self.num_epochs):
            for images, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, batch_labels in val_loader:
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def get_state_dict(self) -> Dict:
        """Get model state dict"""
        return self.model.state_dict()
    
    def  load_state_dict(self, state_dict: Dict):
        """Load model state dict"""
        self.model.load_state_dict(state_dict)
    
    def serialize_state_dict(self, state_dict: Dict) -> bytes:
        """Serialize state dict to bytes"""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        return buffer.read()
    
    def deserialize_state_dict(self, data: bytes) -> Dict:
        """Deserialize state dict from bytes"""
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location='cpu')
