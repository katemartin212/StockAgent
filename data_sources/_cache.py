#!/usr/bin/env python3
"""
_cache.py — Two-layer TTL cache (memory + SQLite) + structured logging.

Layer 1: in-memory dict  — zero latency, cleared on server restart
Layer 2: SQLite file      — microsecond reads, survives server restart

Usage:
    from data_sources._cache import cache_get, cache_set, log_fetch, cache_stats
"""

import time
import json
import sqlite3
import threading
import logging
from logging.handlers import RotatingFileHandler
import os

# ── Logging setup ─────────────────────────────────────────────────────────────

def _make_logger() -> logging.Logger:
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent.log")
    logger = logging.getLogger("stock_agent")
    if not logger.handlers:
        handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=2)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = _make_logger()

# ── SQLite setup ──────────────────────────────────────────────────────────────

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache.db")
_db_lock = threading.Lock()
_db_conn: sqlite3.Connection | None = None

def _get_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _db_conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key      TEXT PRIMARY KEY,
                ts       REAL NOT NULL,
                data     TEXT NOT NULL
            )
        """)
        _db_conn.execute("CREATE INDEX IF NOT EXISTS cache_ts ON cache(ts)")
        _db_conn.commit()
    return _db_conn

# ── In-memory layer ───────────────────────────────────────────────────────────

_STORE: dict[str, tuple[float, dict]] = {}
DEFAULT_TTL = 1800  # 30 minutes

# ── Hit-rate tracking ─────────────────────────────────────────────────────────

_stats = {"mem_hits": 0, "db_hits": 0, "misses": 0, "sets": 0, "saved_ms": 0}

# ── Public API ────────────────────────────────────────────────────────────────

def cache_key(source: str, *args) -> str:
    return f"{source}:" + ":".join(str(a).upper() for a in args)


def cache_get(key: str, ttl: int = DEFAULT_TTL) -> dict | None:
    """
    Check memory first (zero latency), then SQLite (microseconds).
    Returns the cached dict if within TTL, else None.
    """
    now = time.time()

    # Layer 1: memory
    entry = _STORE.get(key)
    if entry and (now - entry[0]) < ttl:
        _stats["mem_hits"] += 1
        elapsed = entry[1].get("_elapsed_ms", 500)
        _stats["saved_ms"] += elapsed
        return entry[1]

    # Layer 2: SQLite
    try:
        with _db_lock:
            db = _get_db()
            row = db.execute("SELECT ts, data FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            ts, data_str = row
            if (now - ts) < ttl:
                data = json.loads(data_str)
                _STORE[key] = (ts, data)   # promote to memory
                _stats["db_hits"] += 1
                elapsed = data.get("_elapsed_ms", 500)
                _stats["saved_ms"] += elapsed
                return data
    except Exception as e:
        logger.warning(f"cache_get SQLite error ({key}): {e}")

    _stats["misses"] += 1
    return None


def cache_set(key: str, value: dict) -> None:
    """Store dict in both memory and SQLite."""
    now = time.time()
    _STORE[key] = (now, value)
    _stats["sets"] += 1
    try:
        data_str = json.dumps(value, default=str)
        with _db_lock:
            db = _get_db()
            db.execute(
                "INSERT OR REPLACE INTO cache (key, ts, data) VALUES (?, ?, ?)",
                (key, now, data_str),
            )
            db.commit()
    except Exception as e:
        logger.warning(f"cache_set SQLite error ({key}): {e}")


def cache_stats() -> dict:
    """Return hit rate and time-saved statistics since server start."""
    total = _stats["mem_hits"] + _stats["db_hits"] + _stats["misses"]
    hit_rate = round((_stats["mem_hits"] + _stats["db_hits"]) / max(total, 1) * 100, 1)
    return {
        "mem_hits":   _stats["mem_hits"],
        "db_hits":    _stats["db_hits"],
        "misses":     _stats["misses"],
        "total":      total,
        "hit_rate_pct": hit_rate,
        "saved_ms":   _stats["saved_ms"],
        "saved_s":    round(_stats["saved_ms"] / 1000, 1),
        "mem_keys":   len(_STORE),
        "db_keys":    _db_key_count(),
    }


def _db_key_count() -> int:
    try:
        with _db_lock:
            db = _get_db()
            return db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    except Exception:
        return 0


def log_fetch(source: str, key: str, cached: bool, elapsed_ms: float = 0.0) -> None:
    """Emit a structured fetch log line."""
    status = "CACHE" if cached else f"LIVE  {elapsed_ms:.0f}ms"
    logger.info(f"{source:<28} {key:<30} {status}")
