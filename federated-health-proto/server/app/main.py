from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.db.database import init_db
from app.model_store.storage import init_storage
from app.api import auth, models, rounds, client_updates, metrics

# Create FastAPI app
app = FastAPI(
    title="Federated Learning Health API",
    description="Production-grade federated learning for healthcare data",
    version="1.0.0"
)

# CORS configuration - allow frontend
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(models.router)
app.include_router(rounds.router)
app.include_router(client_updates.router)
app.include_router(metrics.router)


@app.on_event("startup")
async def startup_event():
    """Initialize database and storage on startup"""
    init_db()
    init_storage()
    print("✓ Database and storage initialized")


@app.get("/")
async def root():
    return {
        "message": "Federated Learning Health API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "auth": "/api/register, /api/login",
            "models": "/api/global-model/{client_type}/latest",
            "rounds": "/api/start-round/{client_type}",
            "updates": "/api/client-update"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
