"""
Simplified storage using local filesystem instead of MinIO.
"""
import os
from pathlib import Path

STORAGE_DIR = Path("./local_storage")

def init_storage():
    """Initialize local file storage"""
    STORAGE_DIR.mkdir(exist_ok=True)
    (STORAGE_DIR / "global_models").mkdir(exist_ok=True)
    (STORAGE_DIR / "client_updates").mkdir(exist_ok=True)
    print("✓ Local storage initialized")

def upload_model(object_name: str, data: bytes):
    """Upload model to local storage"""
    file_path = STORAGE_DIR / object_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(data)
    return str(file_path)

def download_model(object_name: str) -> bytes:
    """Download model from local storage"""
    file_path = STORAGE_DIR / object_name
    if not file_path.exists():
        raise FileNotFoundError(f"Model not found: {object_name}")
    return file_path.read_bytes()

def list_models(prefix: str = "") -> list:
    """List models in storage"""
    search_path = STORAGE_DIR / prefix if prefix else STORAGE_DIR
    if not search_path.exists():
        return []
    return [str(p.relative_to(STORAGE_DIR)) for p in search_path.rglob("*") if p.is_file()]

def delete_model(object_name: str):
    """Delete model from storage"""
    file_path = STORAGE_DIR / object_name
    if file_path.exists():
        file_path.unlink()
