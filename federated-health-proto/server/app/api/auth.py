from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import timedelta

from app.db.database import get_db
from app.db.models import User, ClientType, UserRole
from app.auth.jwt_handler import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password
)

router = APIRouter(prefix="/api", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


# Pydantic schemas
class UserRegister(BaseModel):
    user_id: str
    email: EmailStr
    password: str
    client_type: ClientType
    organization: str
    role: UserRole


class UserLogin(BaseModel):
    user_id: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    client_type: str


class UserResponse(BaseModel):
    id: int
    user_id: str
    email: str
    client_type: ClientType
    organization: str
    role: UserRole

    class Config:
        from_attributes = True


# Dependency to get current user
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.user_id == user_id).first()
    if user is None:
        raise credentials_exception
    
    return user


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user"""
    
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.user_id == user_data.user_id) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID or email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        user_id=user_data.user_id,
        email=user_data.email,
        hashed_password=hashed_password,
        client_type=user_data.client_type,
        organization=user_data.organization,
        role=user_data.role
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Generate access token
    access_token = create_access_token(
        data={"sub": new_user.user_id, "client_type": new_user.client_type.value}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=new_user.user_id,
        client_type=new_user.client_type.value
    )


@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login a user"""
    
    user = db.query(User).filter(User.user_id == user_data.user_id).first()
    
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user ID or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    access_token = create_access_token(
        data={"sub": user.user_id, "client_type": user.client_type.value}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.user_id,
        client_type=user.client_type.value
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user
