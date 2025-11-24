"""
Complete Federated Learning Server - Production Grade
- Automatic rounds every 60 seconds
- LightGBM Distillation (Hospital/Clinic)
- PyTorch FedAvg (Lab/IoT)
- Risk Score Analysis
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import pandas as pd
import numpy as np
from pathlib import Path
import json
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import io
import copy
import os

# --- Configuration ---
ROUND_DURATION = 60  # seconds
MIN_PARTICIPANTS = 1
SECRET_KEY = "fl_demo_secret_key_2024"

# --- Storage Setup ---
STORAGE_DIR = Path("./fl_storage")
STORAGE_DIR.mkdir(exist_ok=True)
(STORAGE_DIR / "datasets").mkdir(exist_ok=True)
(STORAGE_DIR / "models").mkdir(exist_ok=True)
(STORAGE_DIR / "updates").mkdir(exist_ok=True)

USERS_FILE = STORAGE_DIR / "users.json"

# --- Database (JSON) ---
def load_json(path):
    if path.exists():
        with open(path, 'r') as f: return json.load(f)
    return {}

def save_json(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=2)

users_db = load_json(USERS_FILE)
rounds_db = []
client_updates_db = []
datasets_db = {}

# --- Global State for Models ---
# Store global models in memory for quick access, persist to disk
global_models = {
    "hospital": None, # LightGBM Booster
    "clinic": None,   # LightGBM Booster
    "lab": None,      # PyTorch State Dict
    "iot": None       # PyTorch State Dict
}

# Proxy Data for Distillation (Hospital/Clinic)
# Generated on startup
proxy_data = {
    "hospital": None,
    "clinic": None
}

# --- PyTorch Models ---

# Lab: Simple MLP for feature-based classification (simulating "embedding" output of a CNN)
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

# IoT: 1D CNN for time-series
class IoTModel(nn.Module):
    def __init__(self, input_features=10, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # x shape: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# Initialize global PyTorch models
global_models["lab"] = LabModel().state_dict()
global_models["iot"] = IoTModel().state_dict()

# --- Auth ---
# Switch to pbkdf2_sha256 to avoid bcrypt version compatibility issues
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
app = FastAPI(title="FedHealth.AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def generate_proxy_data():
    """Generate synthetic proxy data for distillation"""
    # Hospital Proxy (Heart Failure features)
    # 12 features
    proxy_data["hospital"] = pd.DataFrame(
        np.random.rand(100, 12), 
        columns=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
                 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    )
    
    # Clinic Proxy (Health Status features - Real Data has 3 features)
    # 3 features: pulse, body temperature, SpO2
    proxy_data["clinic"] = pd.DataFrame(
        np.random.rand(100, 3),
        columns=['pulse', 'body temperature', 'SpO2']
    )

def get_current_user(authorization: str):
    if not authorization: raise HTTPException(401, "No auth header")
    token = authorization.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except:
        raise HTTPException(401, "Invalid token")

# --- Core FL Logic ---

async def start_round_timer(client_type: str):
    """Auto-start rounds loop"""
    while True:
        # 1. Create Round
        round_num = len([r for r in rounds_db if r["client_type"] == client_type]) + 1
        round_id = len(rounds_db) + 1
        new_round = {
            "round_id": round_id,
            "round_number": round_num,
            "client_type": client_type,
            "status": "active",
            "num_participants": 0,
            "started_at": datetime.now().isoformat(),
            "window_seconds": ROUND_DURATION,
            "avg_accuracy": 0.0,
            "avg_f1": 0.0
        }
        rounds_db.append(new_round)
        print(f"🔄 [Round {round_num}] Started for {client_type}")

        # 2. Wait
        await asyncio.sleep(ROUND_DURATION)

        # 3. Aggregate
        await aggregate(round_id)

async def aggregate(round_id):
    round_obj = next((r for r in rounds_db if r["round_id"] == round_id), None)
    if not round_obj or round_obj["status"] != "active": return

    updates = [u for u in client_updates_db if u["round_id"] == round_id]
    
    if not updates:
        round_obj["status"] = "completed"
        print(f"⚠️ [Round {round_obj['round_number']}] No participants.")
        return

    client_type = round_obj["client_type"]
    print(f"🧩 [Round {round_obj['round_number']}] Aggregating {len(updates)} updates for {client_type}...")

    # Calculate Metrics
    avg_acc = sum(u["metrics"]["accuracy"] for u in updates) / len(updates)
    avg_f1 = sum(u["metrics"]["f1_score"] for u in updates) / len(updates)
    round_obj["avg_accuracy"] = avg_acc
    round_obj["avg_f1"] = avg_f1

    # Perform Aggregation Strategy
    if client_type in ["hospital", "clinic"]:
        # LightGBM: Prediction Aggregation + Distillation
        await aggregate_lightgbm(client_type, updates, round_obj["round_number"])
    else:
        # PyTorch: FedAvg (Weight Averaging)
        await aggregate_pytorch(client_type, updates, round_obj["round_number"])

    round_obj["status"] = "completed"
    round_obj["num_participants"] = len(updates)
    print(f"✅ [Round {round_obj['round_number']}] Completed. Acc: {avg_acc:.2%}")

async def aggregate_lightgbm(client_type, updates, round_num):
    """
    Distillation:
    1. Load proxy predictions from all clients
    2. Average them (soft labels)
    3. Train a global student model on (Proxy Data, Soft Labels)
    """
    try:
        # Load proxy X
        X_proxy = proxy_data[client_type]
        
        # Collect predictions
        all_preds = []
        for u in updates:
            preds = np.array(u["proxy_predictions"])
            all_preds.append(preds)
        
        # Average predictions (Ensemble)
        avg_preds = np.mean(all_preds, axis=0)
        
        # Train Student Model
        # Binary classification: avg_preds are probabilities
        # We treat them as labels for the student
        
        # Create dataset
        lgb_train = lgb.Dataset(X_proxy, label=avg_preds)
        
        params = {
            'objective': 'regression', # Fitting to probabilities
            'metric': 'mse',
            'num_leaves': 15,
            'max_depth': 4,
            'learning_rate': 0.05,
            'verbose': -1
        }
        
        student_model = lgb.train(params, lgb_train, num_boost_round=50)
        
        # Save Global Model
        save_path = STORAGE_DIR / "models" / f"global_{client_type}_v{round_num}.txt"
        student_model.save_model(str(save_path))
        global_models[client_type] = student_model
        
    except Exception as e:
        print(f"❌ Distillation failed: {e}")

async def aggregate_pytorch(client_type, updates, round_num):
    """
    FedAvg:
    1. Load state_dicts from disk
    2. Weighted average of tensors
    3. Save new global state_dict
    """
    try:
        # Initialize with first update's structure
        first_state = torch.load(updates[0]["model_path"])
        avg_state = copy.deepcopy(first_state)
        
        # Reset accumulators
        for key in avg_state:
            avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
            
        total_samples = sum(u["sample_count"] for u in updates)
        
        for u in updates:
            state = torch.load(u["model_path"])
            weight = u["sample_count"] / total_samples
            for key in state:
                avg_state[key] += state[key] * weight
                
        # Update global model
        global_models[client_type] = avg_state
        
        # Save
        save_path = STORAGE_DIR / "models" / f"global_{client_type}_v{round_num}.pth"
        torch.save(avg_state, str(save_path))
        
    except Exception as e:
        print(f"❌ FedAvg failed: {e}")

# --- API Endpoints ---

@app.on_event("startup")
async def startup():
    generate_proxy_data()
    for c_type in ["hospital", "clinic", "lab", "iot"]:
        asyncio.create_task(start_round_timer(c_type))

@app.post("/api/register")
async def register(user: dict):
    if user["user_id"] in users_db: raise HTTPException(400, "User exists")
    users_db[user["user_id"]] = {
        **user, 
        "password_hash": pwd_context.hash(user["password"]),
        "dataset_uploaded": False
    }
    save_json(USERS_FILE, users_db)
    token = jwt.encode({"sub": user["user_id"], "client_type": user["client_type"]}, SECRET_KEY)
    return {"access_token": token, "client_type": user["client_type"]}

@app.post("/api/login")
async def login(creds: dict):
    user = users_db.get(creds["user_id"])
    if not user or not pwd_context.verify(creds["password"], user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    token = jwt.encode({"sub": user["user_id"], "client_type": user["client_type"]}, SECRET_KEY)
    return {"access_token": token, "client_type": user["client_type"], "user_id": user["user_id"], "organization": user["organization"]}

@app.get("/api/me")
async def me(authorization: str = Header(None)):
    data = get_current_user(authorization)
    user = users_db.get(data["sub"])
    return user

@app.post("/api/upload-dataset")
async def upload(file: UploadFile = File(...), authorization: str = Header(None)):
    data = get_current_user(authorization)
    user_id = data["sub"]
    
    path = STORAGE_DIR / "datasets" / f"{user_id}_{file.filename}"
    content = await file.read()
    path.write_bytes(content)
    
    datasets_db[user_id] = {"path": str(path), "client_type": data["client_type"]}
    users_db[user_id]["dataset_uploaded"] = True
    save_json(USERS_FILE, users_db)
    return {"status": "uploaded"}

@app.post("/api/train")
async def train(authorization: str = Header(None)):
    """
    Server-side 'Local Training' Simulation
    """
    user_data = get_current_user(authorization)
    user_id = user_data["sub"]
    client_type = user_data["client_type"]
    
    if user_id not in datasets_db: raise HTTPException(400, "No dataset")
    
    # Get Active Round
    active_round = next((r for r in rounds_db if r["client_type"] == client_type and r["status"] == "active"), None)
    if not active_round: raise HTTPException(400, "No active round")
    
    # Check duplicate
    if any(u["user_id"] == user_id and u["round_id"] == active_round["round_id"] for u in client_updates_db):
        raise HTTPException(400, "Already participated in this round")
        
    dataset_path = datasets_db[user_id]["path"]
    
    # TRAIN
    if client_type in ["hospital", "clinic"]:
        result = train_lightgbm_local(dataset_path, client_type)
    elif client_type == "lab":
        result = train_lab_local(dataset_path)
    elif client_type == "iot":
        result = train_iot_local(dataset_path)
    else:
        raise HTTPException(400, "Unknown type")
        
    # Save Update
    update_id = len(client_updates_db) + 1
    update_record = {
        "update_id": update_id,
        "round_id": active_round["round_id"],
        "user_id": user_id,
        "client_type": client_type,
        "metrics": result["metrics"],
        "sample_count": result["sample_count"],
        "model_path": result["model_path"],
        "proxy_predictions": result.get("proxy_predictions") # Only for LGBM
    }
    client_updates_db.append(update_record)
    
    return {
        "message": "Training complete", 
        "metrics": result["metrics"],
        "round_id": active_round["round_id"]
    }

# --- Training Logic ---

def train_lightgbm_local(path, client_type):
    df = pd.read_csv(path, comment='#')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    params = {'objective': 'binary', 'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.05, 'verbose': -1}
    model = lgb.train(params, lgb_train, num_boost_round=50, valid_sets=[lgb_val])
    
    # Metrics
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_score": float(f1_score(y_val, y_pred, average='weighted', zero_division=0)),
        "precision": float(precision_score(y_val, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, average='weighted', zero_division=0))
    }
    
    # Distillation: Predict on Proxy
    proxy_preds = model.predict(proxy_data[client_type]).tolist()
    
    # Save Model (optional for LGBM in this scheme, but good for backup)
    model_path = STORAGE_DIR / "updates" / f"lgbm_{datetime.now().timestamp()}.txt"
    model.save_model(str(model_path))
    
    return {
        "metrics": metrics,
        "sample_count": len(df),
        "model_path": str(model_path),
        "proxy_predictions": proxy_preds
    }

def train_lab_local(path):
    # Load Tabular Features (simulating CNN features)
    df = pd.read_csv(path, comment='#')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    
    # Train
    model = LabModel(input_dim=X.shape[1])
    # Load global weights if exist
    if global_models["lab"]:
        model.load_state_dict(global_models["lab"])
        
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(5): # 5 Epochs
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == y).float().mean().item()
        f1 = f1_score(y, preds, average='macro', zero_division=0)
        
    # Save State Dict
    save_path = STORAGE_DIR / "updates" / f"lab_{datetime.now().timestamp()}.pth"
    torch.save(model.state_dict(), str(save_path))
    
    return {
        "metrics": {"accuracy": acc, "f1_score": f1, "precision": acc, "recall": acc},
        "sample_count": len(df),
        "model_path": str(save_path)
    }

def train_iot_local(path):
    # Load Tabular Data (simulating time series)
    df = pd.read_csv(path, comment='#')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    
    model = IoTModel(input_features=X.shape[1])
    if global_models["iot"]:
        model.load_state_dict(global_models["iot"])
        
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == y).float().mean().item()
        f1 = f1_score(y, preds, average='macro', zero_division=0)
        
    save_path = STORAGE_DIR / "updates" / f"iot_{datetime.now().timestamp()}.pth"
    torch.save(model.state_dict(), str(save_path))
    
    return {
        "metrics": {"accuracy": acc, "f1_score": f1, "precision": acc, "recall": acc},
        "sample_count": len(df),
        "model_path": str(save_path)
    }

# --- Info Endpoints ---

@app.get("/api/rounds/{client_type}")
async def get_rounds(client_type: str):
    return [r for r in rounds_db if r["client_type"] == client_type][-10:]

@app.get("/api/metrics/{client_type}")
async def get_metrics(client_type: str):
    completed = [r for r in rounds_db if r["client_type"] == client_type and r["status"] == "completed"]
    return [{"round_number": r["round_number"], "accuracy": r["avg_accuracy"], "f1_score": r["avg_f1"]} for r in completed]

@app.get("/api/download-model/{client_type}")
async def download_model(client_type: str):
    # Find latest
    rounds = [r for r in rounds_db if r["client_type"] == client_type and r["status"] == "completed"]
    if not rounds: raise HTTPException(404, "No model yet")
    latest = rounds[-1]
    
    ext = "txt" if client_type in ["hospital", "clinic"] else "pth"
    path = STORAGE_DIR / "models" / f"global_{client_type}_v{latest['round_number']}.{ext}"
    
    if not path.exists(): raise HTTPException(404, "File missing")
    return FileResponse(str(path), filename=f"global_model.{ext}")

@app.get("/api/risk-score")
async def risk_score(authorization: str = Header(None)):
    user_data = get_current_user(authorization)
    client_type = user_data["client_type"]
    
    # Calculate Risk Score
    # Formula: 100 * (1 - weighted_f1)
    # Get last 3 rounds
    rounds = [r for r in rounds_db if r["client_type"] == client_type and r["status"] == "completed"][-3:]
    if not rounds:
        return {"risk_score": 50, "level": "Unknown", "factors": ["Insufficient data"]}
        
    avg_f1 = sum(r["avg_f1"] for r in rounds) / len(rounds)
    risk_val = 100 * (1 - avg_f1)
    
    level = "Low" if risk_val < 33 else "Medium" if risk_val < 66 else "High"
    
    return {
        "risk_score": round(risk_val, 1),
        "level": level,
        "factors": ["Model Convergence", "Participation Rate"],
        "recommendation": "Increase local epochs" if risk_val > 50 else "Maintain current protocol"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
