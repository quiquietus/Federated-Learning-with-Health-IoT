from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import gzip
import json

from app.db.database import get_db
from app.db.models import User, ClientUpdate, Round, ClientType
from app.api.auth import get_current_user
from app.model_store.storage import upload_model

router = APIRouter(prefix="/api", tags=["client-updates"])


class ClientUpdateRequest(BaseModel):
    round_id: int
    sample_count: int
    client_accuracy: Optional[float] = None
    client_f1_score: Optional[float] = None
    client_precision: Optional[float] = None
    client_recall: Optional[float] = None
    client_loss: Optional[float] = None
    training_time_seconds: Optional[float] = None
    compression_used: bool = False
    delta_update: bool = False


class ClientUpdateResponse(BaseModel):
    update_id: int
    round_id: int
    message: str


@router.post("/client-update", response_model=ClientUpdateResponse)
async def submit_client_update(
    round_id: int = Form(...),
    sample_count: int = Form(...),
    model_update: UploadFile = File(...),
    client_accuracy: Optional[float] = Form(None),
    client_f1_score: Optional[float] = Form(None),
    client_precision: Optional[float] = Form(None),
    client_recall: Optional[float] = Form(None),
    client_loss: Optional[float] = Form(None),
    training_time_seconds: Optional[float] = Form(None),
    compression_used: bool = Form(False),
    delta_update: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit a client model update for aggregation"""
    
    # Verify round exists and is active
    round_obj = db.query(Round).filter(Round.id == round_id).first()
    if not round_obj:
        raise HTTPException(status_code=404, detail="Round not found")
    
    if round_obj.status not in ["active", "pending"]:
        raise HTTPException(status_code=400, detail="Round is not accepting updates")
    
    # Verify client type matches round
    if current_user.client_type != round_obj.client_type:
        raise HTTPException(status_code=403, detail="Client type mismatch")
    
    # Read model update file
    model_data = await model_update.read()
    
    # Store in MinIO
    object_name = f"updates/{round_obj.client_type.value}/round_{round_id}/user_{current_user.id}_{model_update.filename}"
    success = upload_model(object_name, model_data)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store model update")
    
    # Create database record
    client_update = ClientUpdate(
        user_id=current_user.id,
        round_id=round_id,
        model_update_path=object_name,
        sample_count=sample_count,
        client_accuracy=client_accuracy,
        client_f1_score=client_f1_score,
        client_precision=client_precision,
        client_recall=client_recall,
        client_loss=client_loss,
        training_time_seconds=training_time_seconds,
        compression_used=compression_used,
        delta_update=delta_update
    )
    
    db.add(client_update)
    round_obj.num_participants = db.query(ClientUpdate).filter(
        ClientUpdate.round_id == round_id
    ).count() + 1
    db.commit()
    db.refresh(client_update)
    
    return ClientUpdateResponse(
        update_id=client_update.id,
        round_id=round_id,
        message="Update submitted successfully"
    )
