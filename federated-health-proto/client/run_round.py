#!/usr/bin/env python3
"""
Main script to run a federated learning round.
Loads config, trains model, submits update.
"""
import json
import pandas as pd
import numpy as np
import time
from pathlib import Path
import argparse

from fl_client.client import FederatedClient
from fl_client.models.lightgbm_trainer import LightGBMTrainer
from fl_client.models.mobilenet_trainer import MobileNetV2Trainer
from fl_client.models.cnn1d_trainer import CNN1DTrainer


def load_config():
    """Load client configuration"""
    config_file = Path.home() / ".fl_client_config.json"
    if not config_file.exists():
        print("Config not found. Please run register.py first.")
        exit(1)
    return json.loads(config_file.read_text())


def load_dataset(dataset_path: str, client_type: str):
    """Load dataset based on client type"""
    path = Path(dataset_path)
    
    if client_type in ['hospital', 'clinic']:
        # Load CSV for tabular data
        df = pd.read_csv(path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y
    elif client_type == 'lab':
        # Load image paths and labels
        # Format: CSV with columns [image_path, label]
        df = pd.read_csv(path)
        return df['image_path'].tolist(), df['label'].tolist()
    elif client_type == 'iot':
        # Load time-series data
        # Format: numpy arrays X.npy, y.npy
        X = np.load(path / "X.npy")
        y = np.load(path / "y.npy")
        return X, y
    
    return None, None


def run_round(
    dataset_path: str,
    local_epochs: int = 1,
    batch_size: int = 8,
    num_threads: int = 1
):
    """Run one federated learning round"""
    
    # Load config
    config = load_config()
    server_url = config['server_url']
    token = config['token']
    client_type = config['client_type']
    
    print(f"=== Federated Learning Client ===")
    print(f"Client Type: {client_type}")
    print(f"Server: {server_url}\n")
    
    # Initialize client
    client = FederatedClient(server_url, token, client_type)
    
    # Get active round
    round_id = client.get_active_round()
    if round_id is None:
        print("No active round. Waiting...")
        return
    
    print(f"Active Round: {round_id}")
    
    # Download global model
    print("Downloading global model...")
    global_model = client.download_global_model()
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    X, y = load_dataset(dataset_path, client_type)
    sample_count = len(y)
    print(f"Loaded {sample_count} samples")
    
    # Train based on client type
    print(f"\nTraining locally ({local_epochs} epochs)...")
    start_time = time.time()
    
    if client_type in ['hospital', 'clinic']:
        # LightGBM training
        trainer = LightGBMTrainer(num_boost_round=50)
        
        # Load global model if available
        if global_model:
            trainer.load_model(global_model)
        
        metrics = trainer.train(X, y)
        
        # For distillation: generate predictions on proxy set
        # In real implementation, download proxy set from server
        # For now, use a subset of local data as proxy
        proxy_set = X.sample(min(500, len(X)))
        predictions = trainer.predict_on_proxy(proxy_set)
        model_data = trainer.serialize_predictions(predictions)
        
    elif client_type == 'lab':
        # MobileNet training
        trainer = MobileNetV2Trainer(
            batch_size=batch_size,
            num_epochs=local_epochs,
            num_threads=num_threads
        )
        
        # Load global model if available
        if global_model:
            state_dict = trainer.deserialize_state_dict(global_model)
            trainer.load_state_dict(state_dict)
        
        metrics = trainer.train(X, y)
        state_dict = trainer.get_state_dict()
        model_data = trainer.serialize_state_dict(state_dict)
        
    elif client_type == 'iot':
        # 1D-CNN training
        trainer = CNN1DTrainer(
            batch_size=batch_size,
            num_epochs=local_epochs,
            num_threads=num_threads
        )
        
        # Load global model if available
        if global_model:
            state_dict = trainer.deserialize_state_dict(global_model)
            trainer.load_state_dict(state_dict)
        
        metrics = trainer.train(X, y)
        state_dict = trainer.get_state_dict()
        model_data = trainer.serialize_state_dict(state_dict)
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.1f}s")
    print(f"Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Upload update
    print(f"\nUploading model update...")
    success = client.upload_model_update(
        round_id=round_id,
        model_data=model_data,
        sample_count=sample_count,
        metrics=metrics,
        training_time=training_time,
        use_compression=True
    )
    
    if success:
        print(f"\n✓ Round completed successfully!")
    else:
        print(f"\n✗ Failed to upload update")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning round")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    
    args = parser.parse_args()
    
    run_round(
        dataset_path=args.dataset,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        num_threads=args.threads
    )
