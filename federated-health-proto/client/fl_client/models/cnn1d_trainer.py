"""
1D-CNN trainer for time-series/activity data (IoT Device Hubs).
Lightweight architecture for CPU training.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple
import io


class TimeSeriesDataset(Dataset):
    """Dataset for time-series data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    """Tiny 1D-CNN for time-series classification"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 4, sequence_length: int = 100):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN1DTrainer:
    """Trainer for 1D-CNN on time-series data"""
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 4,
        sequence_length: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        num_epochs: int = 3,
        num_threads: int = 1
    ):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set CPU threads
        torch.set_num_threads(num_threads)
        
        # Model
        self.model = CNN1D(input_channels, num_classes, sequence_length)
        self.model.eval()
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Train 1D-CNN on local time-series data.
        
        Args:
            X: Time-series data (samples, channels, sequence_length)
            y: Labels
        
        Returns:
            metrics: Dict with accuracy, F1, precision, recall
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Train
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
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
    
    def load_state_dict(self, state_dict: Dict):
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
