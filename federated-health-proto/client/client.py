import requests
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import io
import gzip

# --- Configuration ---
API_URL = "http://localhost:8000"
CLIENT_ID = "client_1"
CLIENT_TYPE = "hospital" # Default
TOKEN = ""

# --- Models ---
class LabModel(nn.Module):
    def __init__(self, input_dim=7, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class IoTModel(nn.Module):
    def __init__(self, input_features=10, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# --- Client Logic ---

def register(user_id, password, client_type, org):
    """Register and get token"""
    url = f"{API_URL}/api/register"
    data = {"user_id": user_id, "password": password, "client_type": client_type, "organization": org, "email": f"{user_id}@example.com", "role": "admin"}
    try:
        resp = requests.post(url, json=data)
        if resp.status_code == 200:
            return resp.json()["access_token"]
        elif resp.status_code == 400 and "exists" in resp.text:
            # Login instead
            resp = requests.post(f"{API_URL}/api/login", json={"user_id": user_id, "password": password})
            return resp.json()["access_token"]
        else:
            print(f"❌ Registration failed: {resp.text}")
            return None
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def get_active_round(token, client_type):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{API_URL}/api/rounds/{client_type}", headers=headers)
    if resp.status_code == 200:
        rounds = resp.json()
        active = [r for r in rounds if r["status"] == "active"]
        return active[0] if active else None
    return None

def train_lightgbm(dataset_path, proxy_data=None):
    print("🔬 Training LightGBM...")
    df = pd.read_csv(dataset_path, comment='#')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    params = {'objective': 'binary', 'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.05, 'verbose': -1}
    model = lgb.train(params, lgb_train, num_boost_round=50, valid_sets=[lgb_val])
    
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_score": float(f1_score(y_val, y_pred, average='binary', zero_division=0))
    }
    
    # Proxy Predictions for Distillation
    # In a real scenario, we'd fetch proxy data from server. 
    # For this demo client, we simulate it or use a local subset.
    # We'll skip sending proxy preds in this simple client script 
    # and rely on the server-side simulation for the full demo.
    
    return metrics, model

def run_client(user_id, password, client_type, dataset_path):
    print(f"🚀 Starting Client: {user_id} ({client_type})")
    token = register(user_id, password, client_type, "Local Org")
    if not token: return
    
    print("✅ Authenticated. Waiting for rounds...")
    
    while True:
        round_info = get_active_round(token, client_type)
        if round_info:
            print(f"🔔 Round {round_info['round_number']} Active! Starting training...")
            
            # Train
            try:
                if client_type in ["hospital", "clinic"]:
                    metrics, model = train_lightgbm(dataset_path)
                else:
                    print("⚠️ PyTorch client training not fully implemented in this script version.")
                    metrics = {"accuracy": 0.0, "f1_score": 0.0}
                
                print(f"✅ Training Complete. Acc: {metrics['accuracy']:.2%}")
                
                # Upload (Simulated for this script, real upload would use /api/client-update)
                # The server-side /api/train endpoint handles the full flow for the demo.
                headers = {"Authorization": f"Bearer {token}"}
                requests.post(f"{API_URL}/api/train", headers=headers)
                print("📤 Update sent to server.")
                
                # Wait for round to finish
                time.sleep(60)
                
            except Exception as e:
                print(f"❌ Training failed: {e}")
                time.sleep(10)
        else:
            print(".", end="", flush=True)
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    
    run_client(args.user, "password", args.type, args.data)
