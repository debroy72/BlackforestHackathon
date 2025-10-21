from __future__ import annotations

import os
import datetime as dt
import logging
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, constr
from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError

# ----------------------- Logging -----------------------
logger = logging.getLogger("auth_service")
logging.basicConfig(level=logging.INFO)

# ----------------------- Env ---------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env.auth"))

FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGODB_DB", "klimacook")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALG = "HS256"
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "1440"))  # 24h default

# ----------------------- App/CORS ----------------------
app = FastAPI(title="KlimaCook Auth Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------- Mongo -------------------------
client: Optional[AsyncIOMotorClient] = None
DB = None

# ----------------------- Crypto ------------------------
# Use Argon2 (no 72-byte limit; reliable on macOS/Python 3.13)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(sub: str, minutes: int = JWT_EXPIRE_MIN) -> str:
    expire = dt.datetime.utcnow() + dt.timedelta(minutes=minutes)
    payload = {"sub": sub, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        username: Optional[str] = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await DB.users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ----------------------- Models ------------------------
Username = constr(pattern=r"^[a-zA-Z0-9_.-]{3,30}$")  # pydantic v2 uses 'pattern'

class RegisterIn(BaseModel):
    username: Username
    password: str = Field(min_length=8, max_length=256)  # Argon2 handles long/unicode
    # Accept list or string; normalize in route
    allergies: Union[List[str], str] = []

class RegisterOut(BaseModel):
    id: str
    username: str
    allergies: List[str]
    created_at: dt.datetime

class LoginIn(BaseModel):
    username: Username
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: str
    username: str
    allergies: List[str]
    created_at: dt.datetime

# --------------------- Lifecycle -----------------------
@app.on_event("startup")
async def startup():
    global client, DB
    client = AsyncIOMotorClient(MONGO_URI)
    DB = client[MONGO_DB]
    await DB.users.create_index("username", unique=True, sparse=True)
    logger.info(f"[Auth] Mongo connected: {MONGO_URI} db={MONGO_DB}")

    # argon2 self-test
    try:
        test_hash = hash_password("selftest")
        assert verify_password("selftest", test_hash)
        logger.info("[Auth] argon2 backend OK")
    except Exception as e:
        logger.error("[Auth] argon2 test failed: %s", e)

@app.on_event("shutdown")
async def shutdown():
    if client:
        client.close()

# ----------------------- Debug -------------------------
@app.get("/debug/deps")
async def debug_deps():
    try:
        test_hash = hash_password("x")
        ok = verify_password("x", test_hash)
    except Exception as e:
        ok = False
        test_hash = f"ERR: {e}"
    return {
        "mongo_uri": MONGO_URI,
        "db": MONGO_DB,
        "jwt_secret_set": bool(JWT_SECRET),
        "kdf_ok": ok,
        "hash_sample": test_hash if ok else "hash failed",
        "scheme": "argon2",
    }

# ----------------------- Health ------------------------
@app.get("/health")
async def health():
    return {"ok": True, "mongo_db": MONGO_DB}

# ---------------------- Register -----------------------
@app.post("/auth/register_username", response_model=RegisterOut)
async def register(payload: RegisterIn):
    try:
        # normalize allergies into list[str]
        if isinstance(payload.allergies, str):
            allergies = [a.strip().lower() for a in payload.allergies.split(",") if a.strip()]
        else:
            allergies = [a.strip().lower() for a in (payload.allergies or []) if a.strip()]

        doc = {
            "username": payload.username.lower(),
            "password_hash": hash_password(payload.password),
            "allergies": allergies,
            "created_at": dt.datetime.utcnow(),
        }

        res = await DB.users.insert_one(doc)
        return RegisterOut(
            id=str(res.inserted_id),
            username=doc["username"],
            allergies=doc["allergies"],
            created_at=doc["created_at"],
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Username already taken")
    except Exception as e:
        logger.exception("Register failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Register failed: {e}")

# ----------------------- Login -------------------------
@app.post("/auth/login", response_model=TokenOut)
async def login_form(form: OAuth2PasswordRequestForm = Depends()):
    """Form-encoded login (Swagger-friendly)."""
    try:
        username = form.username.lower()
        user = await DB.users.find_one({"username": username})
        if not user or not verify_password(form.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return TokenOut(access_token=create_access_token(sub=username))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Login (form) failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Login failed: {e}")

@app.post("/auth/login_json", response_model=TokenOut)
async def login_json(payload: LoginIn):
    """JSON login for your frontend."""
    try:
        username = payload.username.lower()
        user = await DB.users.find_one({"username": username})
        if not user or not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return TokenOut(access_token=create_access_token(sub=username))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Login (json) failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Login failed: {e}")

# ----------------------- Me ----------------------------
@app.get("/auth/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return UserOut(
        id=str(user["_id"]),
        username=user["username"],
        allergies=user.get("allergies", []),
        created_at=user["created_at"],
    )