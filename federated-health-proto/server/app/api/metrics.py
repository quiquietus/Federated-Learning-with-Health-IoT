from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.db.models import User, Round, ClientType, RiskScore, ClientUpdate
from app.api.auth import get_current_user

router = APIRouter(prefix="/api", tags=["metrics"])


class MetricPoint(BaseModel):
    round_number: int
    accuracy: Optional[float]
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    timestamp: str


class RiskScoreResponse(BaseModel):
    risk_score: float
    risk_category: str
    client_f1: Optional[float]
    global_f1: Optional[float]
    sample_size: Optional[int]
    recommendations: List[str]


@router.get("/metrics/{client_type}", response_model=List[MetricPoint])
async def get_metrics(
    client_type: ClientType,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get aggregated metrics per round for a client type"""
    
    rounds = db.query(Round).filter(
        Round.client_type == client_type,
        Round.status == "completed"
    ).order_by(Round.round_number).all()
    
    return [
        MetricPoint(
            round_number=r.round_number,
            accuracy=r.avg_accuracy,
            f1_score=r.avg_f1_score,
            precision=r.avg_precision,
            recall=r.avg_recall,
            timestamp=r.completed_at.isoformat() if r.completed_at else r.started_at.isoformat()
        )
        for r in rounds
    ]


@router.get("/risk-score", response_model=RiskScoreResponse)
async def get_risk_score(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get risk score for current user (Insurance Analyst feature)"""
    
    # Get the latest round for user's client type
    latest_round = db.query(Round).filter(
        Round.client_type == current_user.client_type,
        Round.status == "completed"
    ).order_by(Round.round_number.desc()).first()
    
    if not latest_round:
        raise HTTPException(status_code=404, detail="No completed rounds found")
    
    # Get user's latest client update
    client_update = db.query(ClientUpdate).filter(
        ClientUpdate.user_id == current_user.id,
        ClientUpdate.round_id == latest_round.id
    ).first()
    
    if not client_update:
        # No participation, return default high risk
        return RiskScoreResponse(
            risk_score=85.0,
            risk_category="High",
            client_f1=None,
            global_f1=latest_round.avg_f1_score,
            sample_size=None,
            recommendations=[
                "No participation in latest round",
                "Submit model updates to reduce risk",
                "Ensure adequate dataset size"
            ]
        )
    
    # Calculate risk score
    client_f1 = client_update.client_f1_score or 0.5
    global_f1 = latest_round.avg_f1_score or 0.5
    sample_size = client_update.sample_count
    
    # Weighted F1 score
    weighted_f1 = 0.7 * client_f1 + 0.3 * global_f1
    risk_score = 100 * max(0, min(1, 1 - weighted_f1))
    
    # Adjust for sample size
    if sample_size < 100:
        risk_score += 10
    
    risk_score = min(100, risk_score)
    
    # Categorize
    if risk_score < 33:
        risk_category = "Low"
    elif risk_score < 66:
        risk_category = "Medium"
    else:
        risk_category = "High"
    
    # Generate recommendations
    recommendations = []
    if client_f1 < 0.7:
        recommendations.append("Low F1 score detected - review model performance")
    if sample_size < 100:
        recommendations.append("Small sample size - collect more labeled data")
    if client_f1 < global_f1 - 0.1:
        recommendations.append("Local model underperforming - check for data quality issues")
    if not recommendations:
        recommendations.append("Model performance is acceptable")
        recommendations.append("Continue regular participation in federated rounds")
    
    return RiskScoreResponse(
        risk_score=round(risk_score, 2),
        risk_category=risk_category,
        client_f1=client_f1,
        global_f1=global_f1,
        sample_size=sample_size,
        recommendations=recommendations
    )
