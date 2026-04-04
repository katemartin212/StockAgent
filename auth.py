#!/usr/bin/env python3
"""
auth.py — JWT authentication and per-user daily quota management.

Environment variables:
    JWT_SECRET      — signing secret for JWTs (set via `fly secrets set`)
    ADMIN_EMAIL     — email address that gets unlimited quota
    DAILY_QUOTA     — requests per user per calendar day (default: 10)
    AUTH_DB_PATH    — SQLite path (default: auth.db; use /data/auth.db on Fly.io)
"""

import os
import sqlite3
import secrets
from datetime import datetime, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

# ── Config ────────────────────────────────────────────────────────────────────

_JWT_SECRET = os.environ.get("JWT_SECRET") or secrets.token_hex(32)
if not os.environ.get("JWT_SECRET"):
    import logging
    logging.getLogger("stock_agent").warning(
        "JWT_SECRET not set — tokens will be invalidated on restart. "
        "Run: fly secrets set JWT_SECRET=$(openssl rand -hex 32)"
    )

_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_DAYS = 30

ADMIN_EMAIL: str = os.environ.get("ADMIN_EMAIL", "").lower().strip()
DAILY_QUOTA: int = int(os.environ.get("DAILY_QUOTA", "10"))
DB_PATH: str = os.environ.get("AUTH_DB_PATH", "auth.db")

def _hash_password(password: str) -> str:
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.checkpw(pw_bytes, hashed.encode("utf-8"))


# ── Database ──────────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS quota_log (
                user_id       INTEGER NOT NULL,
                date          TEXT NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, date)
            );
        """)


# ── User management ───────────────────────────────────────────────────────────

def create_user(email: str, password: str) -> dict:
    """Register a new user. Raises ValueError if email already taken."""
    email = email.lower().strip()
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    pw_hash = _hash_password(password)
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _db() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (email, pw_hash, now),
            )
            return {"id": cur.lastrowid, "email": email}
    except sqlite3.IntegrityError:
        raise ValueError("Email already registered")


def authenticate(email: str, password: str) -> Optional[dict]:
    """Verify email/password. Returns user dict or None."""
    email = email.lower().strip()
    with _db() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
    if not row:
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "email": row["email"]}


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_token(user_id: int, email: str) -> str:
    exp = datetime.now(timezone.utc).timestamp() + 86400 * _JWT_EXPIRE_DAYS
    return jwt.encode(
        {"sub": str(user_id), "email": email, "exp": int(exp)},
        _JWT_SECRET,
        algorithm=_JWT_ALGORITHM,
    )


def verify_token(token: str) -> dict:
    """Returns {id, email} or raises ValueError."""
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return {"id": int(payload["sub"]), "email": payload["email"]}
    except JWTError as e:
        raise ValueError(f"Invalid or expired token: {e}")


# ── Quota ─────────────────────────────────────────────────────────────────────

def is_admin(email: str) -> bool:
    return bool(ADMIN_EMAIL) and email.lower().strip() == ADMIN_EMAIL


def check_and_increment_quota(user_id: int, email: str) -> tuple[bool, int, int]:
    """
    Check quota and increment if allowed.
    Returns (allowed, used_after_increment, limit).
    Admin always returns (True, 0, -1).
    """
    if is_admin(email):
        return True, 0, -1

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _db() as conn:
        conn.execute(
            """INSERT INTO quota_log (user_id, date, request_count)
               VALUES (?, ?, 0)
               ON CONFLICT(user_id, date) DO NOTHING""",
            (user_id, today),
        )
        row = conn.execute(
            "SELECT request_count FROM quota_log WHERE user_id = ? AND date = ?",
            (user_id, today),
        ).fetchone()
        used = row["request_count"] if row else 0

        if used >= DAILY_QUOTA:
            return False, used, DAILY_QUOTA

        conn.execute(
            """UPDATE quota_log SET request_count = request_count + 1
               WHERE user_id = ? AND date = ?""",
            (user_id, today),
        )
        return True, used + 1, DAILY_QUOTA


def get_quota_status(user_id: int, email: str) -> dict:
    """Return current quota state without incrementing."""
    if is_admin(email):
        return {"used": 0, "limit": -1, "remaining": -1, "is_admin": True,
                "resets": "never"}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _db() as conn:
        row = conn.execute(
            "SELECT request_count FROM quota_log WHERE user_id = ? AND date = ?",
            (user_id, today),
        ).fetchone()
    used = row["request_count"] if row else 0
    return {
        "used": used,
        "limit": DAILY_QUOTA,
        "remaining": max(0, DAILY_QUOTA - used),
        "is_admin": False,
        "resets": "midnight UTC",
    }
