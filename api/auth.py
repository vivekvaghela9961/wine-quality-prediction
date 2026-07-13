import os
import datetime
import hashlib
import secrets
import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Load .env file
load_dotenv()

# Configurations
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "super-secret-key-for-jwt-signing-wine-predictions"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer()


def hash_password(password: str, salt: str = None) -> str:
    """Hash password using cryptographically secure PBKDF2 with salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    ).hex()
    return f"{salt}:{pwd_hash}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    try:
        salt, _ = hashed_password.split(":")
        compare_hash = hash_password(plain_password, salt)
        return compare_hash == hashed_password
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: datetime.timedelta = None) -> str:
    """Generate a JWT access token containing user claims."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """Validate JWT bearer token and return username claim."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=401, detail="Invalid token: missing subject claim"
            )
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )
