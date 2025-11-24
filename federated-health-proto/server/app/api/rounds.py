from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.db.models import User, Round, ClientType, ClientUpdate
from app.api.auth import get_current_user

router = APIRouter(prefix="/api", tags=["rounds"])


class RoundStatus(BaseModel):
    round_id: int
    round_number: int
    client_type: str
    status: str
    num_participants: int
    started_at: str
    completed_at: Optional[str]
    avg_accuracy: Optional[float]
    avg_f1_score: Optional[float]
    avg_precision: Optional[float]
    avg_recall: Optional[float]


class CreateRoundRequest(BaseModel):
    client_type: ClientType


@router.post("/start-round/{client_type}", response_model=RoundStatus)
async def start_round(
    client_type: ClientType,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a new federated learning round for a client type"""
    
    # Get the latest round number for this client type
    latest_round = db.query(Round).filter(
        Round.client_type == client_type
    ).order_by(Round.round_number.desc()).first()
    
    next_round_number = 1 if not latest_round else latest_round.round_number + 1
    
    # Check if there's already an active round
    active_round = db.query(Round).filter(
        Round.client_type == client_type,
        Round.status.in_(["pending", "active"])
    ).first()
    
    if active_round:
        raise HTTPException(
            status_code=400,
            detail=f"Round {active_round.round_number} is already active for {client_type.value}"
        )
    
    # Create new round
    new_round = Round(
        round_number=next_round_number,
        client_type=client_type,
        status="active",
        num_participants=0
    )
    
    db.add(new_round)
    db.commit()
    db.refresh(new_round)
    
    return RoundStatus(
        round_id=new_round.id,
        round_number=new_round.round_number,
        client_type=new_round.client_type.value,
        status=new_round.status,
        num_participants=0,
        started_at=new_round.started_at.isoformat(),
        completed_at=None,
        avg_accuracy=None,
        avg_f1_score=None,
        avg_precision=None,
        avg_recall=None  
    )


@router.get("/round-status/{client_type}/{round_number}", response_model=RoundStatus)
async def get_round_status(
    client_type: ClientType,
    round_number: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get status of a specific round"""
    
    round_obj = db.query(Round).filter(
        Round.client_type == client_type,
        Round.round_number == round_number
    ).first()
    
    if not round_obj:
        raise HTTPException(status_code=404, detail="Round not found")
    
    return RoundStatus(
        round_id=round_obj.id,
        round_number=round_obj.round_number,
        client_type=round_obj.client_type.value,
        status=round_obj.status,
        num_participants=round_obj.num_participants,
        started_at=round_obj.started_at.isoformat(),
        completed_at=round_obj.completed_at.isoformat() if round_obj.completed_at else None,
        avg_accuracy=round_obj.avg_accuracy,
        avg_f1_score=round_obj.avg_f1_score,
        avg_precision=round_obj.avg_precision,
        avg_recall=round_obj.avg_recall
    )


@router.get("/rounds/{client_type}", response_model=List[RoundStatus])
async def list_rounds(
    client_type: ClientType,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all rounds for a client type"""
    
    rounds = db.query(Round).filter(
        Round.client_type == client_type
    ).order_by(Round.round_number.desc()).limit(20).all()
    
    return [
        RoundStatus(
            round_id=r.id,
            round_number=r.round_number,
            client_type=r.client_type.value,
            status=r.status,
            num_participants=r.num_participants,
            started_at=r.started_at.isoformat(),
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            avg_accuracy=r.avg_accuracy,
            avg_f1_score=r.avg_f1_score,
            avg_precision=r.avg_precision,
            avg_recall=r.avg_recall
        )
        for r in rounds
    ]
