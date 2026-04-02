from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import timedelta
import hashlib
from typing import Optional
import re
import mysql.connector
from mysql.connector import Error
from .utils.auth import (
    create_access_token, 
    get_user_by_username,
    get_db_connection,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Password hashing context
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)


# ==========================================
# PYDANTIC MODELS
# ==========================================

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100, description="Username (3-100 characters)")
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, max_length=100, description="Password (min 8 characters)")
    name: str = Field(..., min_length=1, max_length=200, description="Full name")
    age: Optional[int] = Field(None, ge=1, le=150, description="Age (1-150)")
    gender: Optional[str] = Field(None, pattern="^[MFO]$", description="Gender: M, F, or O (Other)")  # 🔧 FIXED: regex → pattern
    
    @field_validator('username')  # 🔧 FIXED: @validator → @field_validator
    @classmethod
    def validate_username(cls, v):
        """Validate username format"""
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscore, dot, and hyphen')
        return v.lower()  # Store usernames in lowercase
    
    @field_validator('password')  # 🔧 FIXED: @validator → @field_validator
    @classmethod
    def validate_password(cls, v):
        """Validate password strength"""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @field_validator('gender')  # 🔧 FIXED: @validator → @field_validator
    @classmethod
    def validate_gender(cls, v):
        """Normalize gender to uppercase"""
        if v:
            return v.upper()
        return v

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    name: str
    age: Optional[int]
    gender: Optional[str]

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def check_username_exists(username: str) -> bool:
    """Check if username already exists in database"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = "SELECT COUNT(*) FROM TBL_USERS WHERE USER_NAME = %s"
        cursor.execute(query, (username.lower(),))
        count = cursor.fetchone()[0]
        
        return count > 0
    except Error as e:
        logger.error(f"Database error checking username: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error"
        )
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def check_email_exists(email: str) -> bool:
    """Check if email already exists in database"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = "SELECT COUNT(*) FROM TBL_USERS WHERE EMAIL = %s"
        cursor.execute(query, (email.lower(),))
        count = cursor.fetchone()[0]
        
        return count > 0
    except Error as e:
        logger.error(f"Database error checking email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error"
        )
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def create_user(signup_data: SignupRequest) -> dict:
    """Create new user in database"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Hash the password
        password_hash = hash_password(signup_data.password)
        
        # Insert user
        insert_query = """
            INSERT INTO TBL_USERS 
            (USER_NAME, EMAIL, NAME, AGE, GENDER, PASSWORD_HASH)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            signup_data.username.lower(),
            signup_data.email.lower(),
            signup_data.name,
            signup_data.age,
            signup_data.gender,
            password_hash
        ))
        
        connection.commit()
        user_id = cursor.lastrowid
        
        # Fetch the created user
        select_query = """
            SELECT USER_ID, USER_NAME, EMAIL, NAME, AGE, GENDER
            FROM TBL_USERS 
            WHERE USER_ID = %s
        """
        cursor.execute(select_query, (user_id,))
        user = cursor.fetchone()
        
        logger.info(f"New user created: {signup_data.username} (ID: {user_id})")
        return user
        
    except Error as e:
        if connection:
            connection.rollback()
        logger.error(f"Database error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# ==========================================
# API ENDPOINTS
# ==========================================

@router.post("/signup", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
async def signup(signup_data: SignupRequest):
    """
    User registration endpoint
    Creates new user account with unique username and email
    """
    # Check if username already exists
    if check_username_exists(signup_data.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists. Please choose a different username."
        )
    
    # Check if email already exists
    if check_email_exists(signup_data.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered. Please use a different email or login."
        )
    
    # Create the user
    user = create_user(signup_data)
    
    # Generate access token for automatic login after signup
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user["USER_NAME"],
            "user_id": user["USER_ID"],
            "email": user["EMAIL"]
        },
        expires_delta=access_token_expires
    )
    
    logger.info(f"User {user['USER_NAME']} signed up successfully")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "user_id": user["USER_ID"],
            "username": user["USER_NAME"],
            "email": user["EMAIL"],
            "name": user["NAME"],
            "age": user["AGE"],
            "gender": user["GENDER"]
        }
    }

@router.post("/token", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login endpoint
    Returns JWT access token
    """
    # Fetch user from database
    user = get_user_by_username(form_data.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(form_data.password, user["PASSWORD_HASH"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user["USER_NAME"],
            "user_id": user["USER_ID"],
            "email": user["EMAIL"]
        },
        expires_delta=access_token_expires
    )
    
    logger.info(f"User {user['USER_NAME']} logged in successfully")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "user_id": user["USER_ID"],
            "username": user["USER_NAME"],
            "email": user["EMAIL"],
            "name": user["NAME"],
            "age": user["AGE"],
            "gender": user["GENDER"]
        }
    }

@router.post("/logout")
async def logout():
    """
    Logout endpoint (client should remove token)
    """
    return {"message": "Successfully logged out"}

@router.get("/check-username/{username}")
async def check_username_availability(username: str):
    """
    Check if username is available
    Useful for real-time validation in frontend
    """
    if len(username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters"
        )
    
    exists = check_username_exists(username)
    
    return {
        "username": username,
        "available": not exists,
        "message": "Username is available" if not exists else "Username already taken"
    }

@router.get("/check-email/{email}")
async def check_email_availability(email: str):
    """
    Check if email is available
    Useful for real-time validation in frontend
    """
    exists = check_email_exists(email)
    
    return {
        "email": email,
        "available": not exists,
        "message": "Email is available" if not exists else "Email already registered"
    }
