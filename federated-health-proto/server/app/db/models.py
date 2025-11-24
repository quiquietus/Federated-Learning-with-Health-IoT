from typing import Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum as SQLEnum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class ClientType(str, enum.Enum):
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    LAB = "lab"
    IOT = "iot"


class UserRole(str, enum.Enum):
    DOCTOR = "doctor"
    INSURANCE_ANALYST = "insurance_analyst"
    OTHER = "other"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    client_type = Column(SQLEnum(ClientType), nullable=False)
    organization = Column(String)
    role = Column(SQLEnum(UserRole), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    client_updates = relationship("ClientUpdate", back_populates="user")
    datasets = relationship("Dataset", back_populates="user")


class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, nullable=False)
    client_type = Column(SQLEnum(ClientType), nullable=False)
    status = Column(String, default="pending")  # pending, active, aggregating, completed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    num_participants = Column(Integer, default=0)
    global_model_path = Column(String, nullable=True)
    
    # Aggregated metrics
    avg_accuracy = Column(Float, nullable=True)
    avg_f1_score = Column(Float, nullable=True)
    avg_precision = Column(Float, nullable=True)
    avg_recall = Column(Float, nullable=True)

    # Relationships
    client_updates = relationship("ClientUpdate", back_populates="round")


class ClientUpdate(Base):
    __tablename__ = "client_updates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=False)
    
    # Model update information
    model_update_path = Column(String, nullable=False)  # Path in MinIO
    sample_count = Column(Integer, nullable=False)
    
    # Client-side metrics
    client_accuracy = Column(Float)
    client_f1_score = Column(Float)
    client_precision = Column(Float)
    client_recall = Column(Float)
    client_loss = Column(Float, nullable=True)
    
    # Metadata
    submitted_at = Column(DateTime, default=datetime.utcnow)
    training_time_seconds = Column(Float, nullable=True)
    compression_used = Column(Boolean, default=False)
    delta_update = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="client_updates")
    round = relationship("Round", back_populates="client_updates")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_name = Column(String, nullable=False)
    dataset_path = Column(String, nullable=False)  # Local path or MinIO path
    upload_type = Column(String, default="local")  # local or server
    num_samples = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="datasets")


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=False)
    
    # Risk calculation
    risk_score = Column(Float, nullable=False)  # 0-100
    risk_category = Column(String, nullable=False)  # Low, Medium, High
    
    # Contributing factors
    client_f1 = Column(Float)
    global_f1 = Column(Float)
    sample_size = Column(Integer)
    
    # Recommendations (stored as JSON array)
    recommendations = Column(JSON, nullable=True)
    
    calculated_at = Column(DateTime, default=datetime.utcnow)
