"""
LightGBM distillation-based aggregation for tabular data (Hospitals, Clinics).
Uses prediction aggregation + soft-label distillation approach.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import gzip
import io
from typing import List, Dict, Tuple, Optional
from app.model_store.storage import download_model


class LightGBMDistiller:
    """Distillation-based aggregator for LightGBM models"""
    
    def __init__(self, proxy_set: Optional[pd.DataFrame] = None, num_classes: int = 2):
        """
        Args:
            proxy_set: Small representative dataset for predictions (500 rows)
            num_classes: Number of classes for classification
        """
        self.proxy_set = proxy_set
        self.num_classes = num_classes
    
    def load_client_predictions(
        self,
        update_paths: List[str],
        sample_counts: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Load prediction probabilities from client updates.
        Clients send their model's predictions on the proxy set.
        
        Returns:
            predictions: Array of shape (n_clients, n_samples, n_classes)
            sample_counts: List of sample counts
        """
        predictions = []
        
        for path in update_paths:
            data = download_model(path)
            if data is None:
                raise ValueError(f"Failed to download from {path}")
            
            try:
                # Try decompression
                data = gzip.decompress(data)
            except:
                pass
            
            # Load predictions (numpy array or pickle)
            buffer = io.BytesIO(data)
            try:
                preds = np.load(buffer, allow_pickle=True)
            except:
                buffer.seek(0)
                preds = pickle.load(buffer)
            
            predictions.append(preds)
        
        return np.array(predictions), sample_counts
    
    def aggregate_predictions(
        self,
        predictions: np.ndarray,
        sample_counts: List[int]
    ) -> np.ndarray:
        """
        Aggregate predictions using weighted averaging.
        
        Args:
            predictions: Shape (n_clients, n_samples, n_classes)
            sample_counts: Weights for each client
        
        Returns:
            aggregated: Shape (n_samples, n_classes)
        """
        total_samples = sum(sample_counts)
        weights = np.array(sample_counts) / total_samples
        
        # Weighted average across clients
        aggregated = np.sum(
            predictions * weights[:, None, None],
            axis=0
        )
        
        return aggregated
    
    def distill_to_lightgbm(
        self,
        proxy_features: pd.DataFrame,
        soft_labels: np.ndarray,
        params: Optional[Dict] = None
    ) -> lgb.Booster:
        """
        Train a new LightGBM model to match the aggregated soft labels.
        
        Args:
            proxy_features: Proxy dataset features
            soft_labels: Aggregated prediction probabilities
            params: LightGBM parameters
        
        Returns:
            Trained LightGBM booster
        """
        if params is None:
            params = {
                'objective': 'multiclass' if self.num_classes > 2 else 'binary',
                'num_class': self.num_classes if self.num_classes > 2 else None,
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 4,
                'learning_rate': 0.05,
                'verbose': -1
            }
        
        # Convert soft labels to class labels for LightGBM
        # Use argmax for hard labels (can also use soft targets with custom objective)
        if self.num_classes > 2:
            hard_labels = np.argmax(soft_labels, axis=1)
        else:
            hard_labels = (soft_labels[:, 1] > 0.5).astype(int)
        
        # Create LightGBM dataset
        lgb_train = lgb.Dataset(proxy_features, label=hard_labels)
        
        # Train model
        num_round = 100
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_round,
            valid_sets=[lgb_train],
            valid_names=['train'],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        return booster
    
    def save_booster(self, booster: lgb.Booster) -> bytes:
        """Save LightGBM booster to bytes"""
        buffer = io.BytesIO()
        pickle.dump(booster, buffer)
        buffer.seek(0)
        return buffer.read()


def aggregate_lightgbm_models(
    update_paths: List[str],
    sample_counts: List[int],
    proxy_set: pd.DataFrame,
    num_classes: int = 2
) -> Tuple[bytes, Dict[str, float]]:
    """
    Main aggregation function for LightGBM models.
    
    Args:
        update_paths: Paths to client prediction files
        sample_counts: Sample counts for weighting
        proxy_set: Proxy dataset for distillation
        num_classes: Number of classes
    
    Returns:
        model_bytes: Serialized global LightGBM model
        metrics: Aggregation metrics
    """
    distiller = LightGBMDistiller(proxy_set, num_classes)
    
    # Load predictions from all clients
    predictions, counts = distiller.load_client_predictions(update_paths, sample_counts)
    
    # Aggregate predictions
    aggregated_probs = distiller.aggregate_predictions(predictions, counts)
    
    # Distill to new LightGBM model
    global_booster = distiller.distill_to_lightgbm(proxy_set, aggregated_probs)
    
    # Serialize model
    model_bytes = distiller.save_booster(global_booster)
    
    metrics = {
        "num_clients": len(update_paths),
        "total_samples": sum(sample_counts),
        "proxy_set_size": len(proxy_set),
        "num_classes": num_classes
    }
    
    return model_bytes, metrics
