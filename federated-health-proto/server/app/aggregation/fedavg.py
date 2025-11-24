"""
FedAvg (Federated Averaging) implementation for PyTorch models.
Used for Image (MobileNetV2) and Time-series (1D-CNN) models.
"""
import torch
import gzip
import io
from typing import List, Dict, Tuple
from app.model_store.storage import download_model


def decompress_update(compressed_data: bytes) -> bytes:
    """Decompress gzipped model update"""
    try:
        return gzip.decompress(compressed_data)
    except:
        # If not compressed, return as is
        return compressed_data


def load_state_dict_from_bytes(data: bytes) -> Dict[str, torch.Tensor]:
    """Load PyTorch state dict from bytes"""
    buffer = io.BytesIO(data)
    state_dict = torch.load(buffer, map_location='cpu')
    return state_dict


def federated_averaging(
    update_paths: List[str],
    sample_counts: List[int],
    compression_flags: List[bool]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Perform federated averaging on PyTorch model updates.
    
    Args:
        update_paths: List of MinIO paths to model updates
        sample_counts: List of sample counts for each client
        compression_flags: List indicating if each update is compressed
    
    Returns:
        aggregated_state_dict: Weighted average of all state dicts
        metrics: Aggregation metrics
    """
    
    total_samples = sum(sample_counts)
    
    if total_samples == 0:
        raise ValueError("Total sample count is zero")
    
    # Download and load all updates
    state_dicts = []
    for path, compressed in zip(update_paths, compression_flags):
        data = download_model(path)
        if data is None:
            raise ValueError(f"Failed to download update from {path}")
        
        if compressed:
            data = decompress_update(data)
        
        state_dict = load_state_dict_from_bytes(data)
        state_dicts.append(state_dict)
    
    # Verify all state dicts have same keys
    reference_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if set(sd.keys()) != reference_keys:
            raise ValueError("State dict keys mismatch between clients")
    
    # Weighted averaging
    aggregated = {}
    for key in reference_keys:
        weighted_sum = sum(
            state_dict[key] * (count / total_samples)
            for state_dict, count in zip(state_dicts, sample_counts)
        )
        aggregated[key] = weighted_sum
    
    metrics = {
        "num_clients": len(state_dicts),
        "total_samples": total_samples,
        "avg_samples_per_client": total_samples / len(state_dicts)
    }
    
    return aggregated, metrics


def save_state_dict_to_bytes(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """Save PyTorch state dict to bytes"""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return buffer.read()


def apply_delta_update(
    global_state_dict: Dict[str, torch.Tensor],
    delta_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Apply delta update to global model"""
    updated = {}
    for key in global_state_dict.keys():
        if key in delta_state_dict:
            updated[key] = global_state_dict[key] + delta_state_dict[key]
        else:
            updated[key] = global_state_dict[key]
    return updated
