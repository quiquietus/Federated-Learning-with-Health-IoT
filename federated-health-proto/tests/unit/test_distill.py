"""
Unit tests for LightGBM distillation.
"""
import pytest
import numpy as np
import pandas as pd
from app.aggregation.distill import LightGBMDistiller


def test_prediction_aggregation():
    """Test aggregation of client predictions"""
    
    distiller = LightGBMDistiller(num_classes=2)
    
    # 2 clients, 100 samples, 2 classes
    predictions = np.array([
        [[0.7, 0.3]] * 100,  # Client 1 predictions
        [[0.3, 0.7]] * 100   # Client 2 predictions
    ])
    
    sample_counts = [100, 100]  # Equal weights
    
    aggregated = distiller.aggregate_predictions(predictions, sample_counts)
    
    # Expected: average of [0.7, 0.3] and [0.3, 0.7] = [0.5, 0.5]
    assert aggregated.shape == (100, 2)
    assert np.allclose(aggregated[0], [0.5, 0.5])


def test_weighted_aggregation():
    """Test weighted aggregation"""
    
    distiller = LightGBMDistiller(num_classes=2)
    
    predictions = np.array([
        [[0.8, 0.2]] * 10,  # Client 1: 10 samples
        [[0.2, 0.8]] * 30   # Client 2: 30 samples  
    ])
    
    sample_counts = [10, 30]
    
    aggregated = distiller.aggregate_predictions(predictions, sample_counts)
    
    # Expected: 0.25 * [0.8, 0.2] + 0.75 * [0.2, 0.8] = [0.35, 0.65]
    expected = np.array([0.35, 0.65])
    assert np.allclose(aggregated[0], expected, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
