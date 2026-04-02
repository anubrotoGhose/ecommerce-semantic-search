from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# MySQL Configuration
MYSQL_HOST = os.getenv("mysql_um_db_host", "localhost")
MYSQL_PORT = int(os.getenv("mysql_um_db_port", 3306))
MYSQL_USER = os.getenv("mysql_um_db_user", "root")
MYSQL_PASSWORD = os.getenv("mysql_um_db_password", "")
MYSQL_DATABASE = os.getenv("mysql_um_db_name", "USER_MANAGEMENT")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_db_connection():
    """Create MySQL database connection"""
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        return connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )

# ==========================================
# TOKEN FUNCTIONS
# ==========================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError as e:
        logger.error(f"JWT verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ==========================================
# USER FUNCTIONS
# ==========================================

def get_user_by_username(username: str):
    """Fetch user from database by username"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
            SELECT USER_ID, USER_NAME, EMAIL, NAME, AGE, GENDER, PASSWORD_HASH
            FROM TBL_USERS 
            WHERE USER_NAME = %s
        """
        cursor.execute(query, (username,))
        user = cursor.fetchone()
        
        return user
    except Error as e:
        logger.error(f"Database query error: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def get_user_by_id(user_id: int):
    """Fetch user from database by user_id"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
            SELECT USER_ID, USER_NAME, EMAIL, NAME, AGE, GENDER
            FROM TBL_USERS 
            WHERE USER_ID = %s
        """
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()
        
        return user
    except Error as e:
        logger.error(f"Database query error: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def log_user_query(user_id: int, query: str):
    """Log user search query to database"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        insert_query = """
            INSERT INTO TBL_USER_QUERIES (USER_ID, INPUT_QUERY)
            VALUES (%s, %s)
        """
        cursor.execute(insert_query, (user_id, query))
        connection.commit()
        
        logger.info(f"Query logged for user {user_id}")
    except Error as e:
        logger.error(f"Failed to log query: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# ==========================================
# AUTHENTICATION DEPENDENCIES
# ==========================================

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency to get current authenticated user from JWT token
    Usage: current_user = Depends(get_current_user)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token and extract payload
        payload = verify_token(token)
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if username is None or user_id is None:
            raise credentials_exception
        
        # Fetch user from database
        user = get_user_by_id(user_id)
        
        if user is None:
            raise credentials_exception
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """
    Dependency to ensure user is active (can add additional checks)
    Usage: active_user = Depends(get_current_active_user)
    """
    # Add additional checks here if needed (e.g., user.is_active, user.is_verified)
    return current_user

# ==========================================
# OPTIONAL: Get user but don't require authentication
# ==========================================

async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)):
    """
    Optional authentication - returns user if token is valid, None otherwise
    Usage: optional_user = Depends(get_optional_user)
    """
    if not token:
        return None
    
    try:
        payload = verify_token(token)
        user_id: int = payload.get("user_id")
        if user_id:
            return get_user_by_id(user_id)
    except:
        pass
    
    return None
