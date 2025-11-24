from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import io

from app.db.database import get_db
from app.db.models import User, ClientType, Round
from app.api.auth import get_current_user
from app.model_store.storage import download_model, list_models, upload_model

router = APIRouter(prefix="/api", tags=["models"])


class GlobalModelInfo(BaseModel):
    model_path: str
    round_number: int
    client_type: str
    created_at: str
    avg_accuracy: Optional[float]
    avg_f1_score: Optional[float]


@router.get("/global-model/{client_type}/latest")
async def get_latest_global_model(
    client_type: ClientType,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download the latest global model for a specific client type"""
    
    # Check if user's client type matches requested type
    if current_user.client_type != client_type:
        raise HTTPException(status_code=403, detail="Access denied to this model type")
    
    # Get latest completed round for this client type
    latest_round = db.query(Round).filter(
        Round.client_type == client_type,
        Round.status == "completed",
        Round.global_model_path.isnot(None)
    ).order_by(Round.round_number.desc()).first()
    
    if not latest_round:
        raise HTTPException(status_code=404, detail="No global model available yet")
    
    # Download model from MinIO
    model_data = download_model(latest_round.global_model_path)
    
    if model_data is None:
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Determine file extension based on client type
    if client_type in [ClientType.HOSPITAL, ClientType.CLINIC]:
        filename = f"global_model_v{latest_round.round_number}.txt"
        media_type = "text/plain"
    else:
        filename = f"global_model_v{latest_round.round_number}.pth"
        media_type = "application/octet-stream"
    
    return StreamingResponse(
        io.BytesIO(model_data),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/global-model/{client_type}/list", response_model=List[GlobalModelInfo])
async def list_global_models(
    client_type: ClientType,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all available global models for a client type"""
    
    if current_user.client_type != client_type:
        raise HTTPException(status_code=403, detail="Access denied to this model type")
    
    rounds = db.query(Round).filter(
        Round.client_type == client_type,
        Round.status == "completed",
        Round.global_model_path.isnot(None)
    ).order_by(Round.round_number.desc()).all()
    
    return [
        GlobalModelInfo(
            model_path=r.global_model_path,
            round_number=r.round_number,
            client_type=r.client_type.value,
            created_at=r.completed_at.isoformat() if r.completed_at else "",
            avg_accuracy=r.avg_accuracy,
            avg_f1_score=r.avg_f1_score
        )
        for r in rounds
    ]


@router.get("/download-final-model/{client_id}")
async def download_final_model(
    client_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download the final trained model for a specific client"""
    
    # For now, return the latest global model
    # In future, this could return client-specific fine-tuned models
    return await get_latest_global_model(current_user.client_type, current_user, db)
