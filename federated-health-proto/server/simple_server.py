"""
Standalone simplified server for demo - runs without Docker/Postgres/Redis/MinIO
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import jwt  # PyJWT package is imported as 'jwt'
from datetime import datetime, timedelta
from passlib.context import CryptContext
import json
from pathlib import Path

# Simple in-memory storage
users_db = {}
rounds_db = []
metrics_db = []

# JWT setup
SECRET_KEY = "demo_secret_key_123"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="Federated Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserRegister(BaseModel):
    user_id: str
    email: str
    password: str
    client_type: str
    organization: str
    role: str = "doctor"

class UserLogin(BaseModel):
    user_id: str
    password: str

# Helper functions
def create_token(user_id: str, client_type: str):
    exp = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode({"sub": user_id, "client_type": client_type, "exp": exp}, SECRET_KEY, algorithm="HS256")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except:
        return None

# Routes
@app.get("/")
async def root():
    return {"message": "Federated Learning API", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/register")
async def register(user: UserRegister):
    if user.user_id in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    
    users_db[user.user_id] = {
        "user_id": user.user_id,
        "email": user.email,
        "password_hash": pwd_context.hash(user.password),
        "client_type": user.client_type,
        "organization": user.organization,
        "role": user.role
    }
    
    token = create_token(user.user_id, user.client_type)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.user_id,
        "client_type": user.client_type
    }

@app.post("/api/login")
async def login(creds: UserLogin):
    user = users_db.get(creds.user_id)
    if not user or not pwd_context.verify(creds.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user["user_id"], user["client_type"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user["user_id"],
        "client_type": user["client_type"]
    }

@app.post("/api/start-round/{client_type}")
async def start_round(client_type: str):
    round_num = len([r for r in rounds_db if r["client_type"] == client_type]) + 1
    new_round = {
        "round_id": len(rounds_db) + 1,
        "round_number": round_num,
        "client_type": client_type,
        "status": "active",
        "num_participants": 0,
        "started_at": datetime.now ().isoformat(),
        "avg_accuracy": None,
        "avg_f1_score": None
    }
    rounds_db.append(new_round)
    return new_round

@app.get("/api/rounds/{client_type}")
async def get_rounds(client_type: str):
    return [r for r in rounds_db if r["client_type"] == client_type]

@app.get("/api/metrics/{client_type}")
async def get_metrics(client_type: str):
    # Return dummy metrics for demo
    completed_rounds = [r for r in rounds_db if r["client_type"] == client_type and r["status"] == "completed"]
    return [
        {
            "round_number": r["round_number"],
            "accuracy": r.get("avg_accuracy", 0.85),
            "f1_score": r.get("avg_f1_score", 0.82),
            "precision": 0.83,
            "recall": 0.81
        }
        for r in completed_rounds
    ]

@app.get("/api/risk-score")
async def get_risk_score():
    return {
        "risk_score": 65.5,
        "risk_level": "Medium",
        "factors": ["Model performance variance", "Low participation rate"],
        "recommendations": ["Increase client participation", "Monitor data quality"]
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("Starting Federated Learning Server (Simplified)")
    print("=" * 50)
    print("API: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
