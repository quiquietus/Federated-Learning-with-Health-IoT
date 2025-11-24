"""
Celery worker for background aggregation tasks.
Monitors rounds and triggers aggregation when conditions are met.
"""
from celery import Celery
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

from app.db.database import SessionLocal
from app.db.models import Round, ClientUpdate, ClientType
from app.aggregation.fedavg import federated_averaging, save_state_dict_to_bytes
from app.aggregation.distill import aggregate_lightgbm_models
from app.model_store.storage import upload_model

# Initialize Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery('worker', broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Configuration
ROUND_WINDOW_SECONDS = int(os.getenv("ROUND_WINDOW_SECONDS", "200"))
MIN_CLIENTS_PER_ROUND = int(os.getenv("MIN_CLIENTS_PER_ROUND", "2"))


def get_proxy_dataset(client_type: ClientType) -> pd.DataFrame:
    """
    Load or generate proxy dataset for LightGBM distillation.
    In production, this should be a carefully curated representative sample.
    For now, we'll generate synthetic data.
    """
    proxy_size = int(os.getenv("PROXY_SET_SIZE", "500"))
    
    if client_type == ClientType.HOSPITAL:
        # Heart failure dataset has 13 features
        return pd.DataFrame({
            'age': np.random.randint(40, 95, proxy_size),
            'anaemia': np.random.randint(0, 2, proxy_size),
            'creatinine_phosphokinase': np.random.randint(23, 7861, proxy_size),
            'diabetes': np.random.randint(0, 2, proxy_size),
            'ejection_fraction': np.random.randint(14, 80, proxy_size),
            'high_blood_pressure': np.random.randint(0, 2, proxy_size),
            'platelets': np.random.uniform(25000, 850000, proxy_size),
            'serum_creatinine': np.random.uniform(0.5, 9.4, proxy_size),
            'serum_sodium': np.random.randint(113, 148, proxy_size),
            'sex': np.random.randint(0, 2, proxy_size),
            'smoking': np.random.randint(0, 2, proxy_size),
            'time': np.random.randint(4, 285, proxy_size),
        })
    elif client_type == ClientType.CLINIC:
        # Health status dataset - simplified
        return pd.DataFrame({
            'age': np.random.randint(18, 80, proxy_size),
            'bmi': np.random.uniform(15, 45, proxy_size),
            'sleep_hours': np.random.uniform(4, 12, proxy_size),
            'exercise_hours': np.random.uniform(0, 10, proxy_size),
            'stress_level': np.random.randint(1, 11, proxy_size),
        })
    
    return pd.DataFrame()


@celery_app.task(name='check_and_aggregate_round')
def check_and_aggregate_round(round_id: int):
    """
    Check if a round is ready for aggregation and perform it.
    Called periodically or when a client update is received.
    """
    
    db = SessionLocal()
    try:
        round_obj = db.query(Round).filter(Round.id == round_id).first()
        
        if not round_obj or round_obj.status != "active":
            return {"status": "skip", "reason": "Round not active"}
        
        # Check conditions
        time_elapsed = (datetime.utcnow() - round_obj.started_at).total_seconds()
        num_updates = db.query(ClientUpdate).filter(ClientUpdate.round_id == round_id).count()
        
        # Aggregate if: enough time passed OR minimum clients reached
        should_aggregate = (
            time_elapsed >= ROUND_WINDOW_SECONDS or
            num_updates >= MIN_CLIENTS_PER_ROUND
        )
        
        if not should_aggregate:
            return {"status": "waiting", "num_updates": num_updates, "time_elapsed": time_elapsed}
        
        if num_updates == 0:
            # No updates, mark as completed without aggregation
            round_obj.status = "completed"
            round_obj.completed_at = datetime.utcnow()
            db.commit()
            return {"status": "completed_no_updates"}
        
        # Mark as aggregating
        round_obj.status = "aggregating"
        db.commit()
        
        # Get all client updates
        updates = db.query(ClientUpdate).filter(ClientUpdate.round_id == round_id).all()
        
        update_paths = [u.model_update_path for u in updates]
        sample_counts = [u.sample_count for u in updates]
        compression_flags = [u.compression_used for u in updates]
        
        # Aggregate based on client type
        client_type = round_obj.client_type
        
        if client_type in [ClientType.HOSPITAL, ClientType.CLINIC]:
            # LightGBM distillation
            proxy_set = get_proxy_dataset(client_type)
            model_bytes, metrics = aggregate_lightgbm_models(
                update_paths,
                sample_counts,
                proxy_set,
                num_classes=2
            )
            model_filename = f"global_model_{client_type.value}_v{round_obj.round_number}.pkl"
        else:
            # PyTorch FedAvg
            aggregated_state_dict, metrics = federated_averaging(
                update_paths,
                sample_counts,
                compression_flags
            )
            model_bytes = save_state_dict_to_bytes(aggregated_state_dict)
            model_filename = f"global_model_{client_type.value}_v{round_obj.round_number}.pth"
        
        # Upload to MinIO
        model_path = f"global_models/{client_type.value}/{model_filename}"
        upload_model(model_path, model_bytes)
        
        # Calculate average metrics
        avg_accuracy = sum(u.client_accuracy for u in updates if u.client_accuracy) / len(updates)
        avg_f1 = sum(u.client_f1_score for u in updates if u.client_f1_score) / len(updates)
        avg_precision = sum(u.client_precision for u in updates if u.client_precision) / len(updates)
        avg_recall = sum(u.client_recall for u in updates if u.client_recall) / len(updates)
        
        # Update round
        round_obj.status = "completed"
        round_obj.completed_at = datetime.utcnow()
        round_obj.global_model_path = model_path
        round_obj.avg_accuracy = avg_accuracy
        round_obj.avg_f1_score = avg_f1
        round_obj.avg_precision = avg_precision
        round_obj.avg_recall = avg_recall
        
        db.commit()
        
        return {
            "status": "completed",
            "round_number": round_obj.round_number,
            "num_clients": metrics['num_clients'],
            "model_path": model_path
        }
        
    except Exception as e:
        db.rollback()
        # Mark round as failed
        if round_obj:
            round_obj.status = "failed"
            db.commit()
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


@celery_app.task(name='periodic_round_check')
def periodic_round_check():
    """Periodically check all active rounds"""
    db = SessionLocal()
    try:
        active_rounds = db.query(Round).filter(Round.status == "active").all()
        for round_obj in active_rounds:
            check_and_aggregate_round.delay(round_obj.id)
    finally:
        db.close()


# Setup periodic tasks
celery_app.conf.beat_schedule = {
    'check-rounds-every-30-seconds': {
        'task': 'periodic_round_check',
        'schedule': 30.0,
    },
}
