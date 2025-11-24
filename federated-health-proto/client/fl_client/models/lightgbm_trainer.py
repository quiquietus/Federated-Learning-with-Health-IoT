"""
LightGBM trainer for tabular data (Hospitals and Clinics).
Optimized for weak CPU hardware.
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple
import pickle
import io


class LightGBMTrainer:
    """Trainer for LightGBM models on tabular healthcare data"""
    
    def __init__(
        self,
        num_boost_round: int = 50,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        num_leaves: int = 15
    ):
        self.params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'verbose': -1
        }
        self.num_boost_round = num_boost_round
        self.model = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train LightGBM model on local data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction for validation
        
        Returns:
            metrics: Dict with accuracy, F1, precision, recall
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Create LightGBM datasets
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        # Train
        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=[lgb_val],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        # Evaluate
        y_pred_proba = self.model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred, average='binary'),
            'precision': precision_score(y_val, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='binary', zero_division=0)
        }
        
        return metrics
    
    def predict_on_proxy(self, proxy_set: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions on proxy dataset for distillation.
        
        Args:
            proxy_set: Proxy features
        
        Returns:
            predictions: Probability predictions (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get probabilities
        proba = self.model.predict(proxy_set)
        
        # Convert to binary class probabilities
        predictions = np.column_stack([1 - proba, proba])
        
        return predictions
    
    def serialize_model(self) -> bytes:
        """Serialize model to bytes"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        buffer = io.BytesIO()
        pickle.dump(self.model, buffer)
        buffer.seek(0)
        return buffer.read()
    
    def serialize_predictions(self, predictions: np.ndarray) -> bytes:
        """Serialize predictions for upload"""
        buffer = io.BytesIO()
        np.save(buffer, predictions)
        buffer.seek(0)
        return buffer.read()
    
    def load_model(self, model_bytes: bytes):
        """Load model from bytes"""
        buffer = io.BytesIO(model_bytes)
        self.model = pickle.load(buffer)
