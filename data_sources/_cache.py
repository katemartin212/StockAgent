#!/usr/bin/env python3
"""
_cache.py — shared in-memory TTL cache + structured logging for all data_sources modules.

Usage:
    from data_sources._cache import cache_get, cache_set, log_fetch
"""

import time
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

# ── In-memory cache ───────────────────────────────────────────────────────────
# key → (timestamp_float, data_dict)

_STORE: dict[str, tuple[float, dict]] = {}
DEFAULT_TTL = 1800  # 30 minutes


def cache_key(source: str, *args) -> str:
    return f"{source}:" + ":".join(str(a).upper() for a in args)


def cache_get(key: str, ttl: int = DEFAULT_TTL) -> dict | None:
    """Return cached dict if within TTL, else None."""
    entry = _STORE.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def cache_set(key: str, value: dict) -> None:
    """Store dict with current timestamp."""
    _STORE[key] = (time.time(), value)


def log_fetch(source: str, key: str, cached: bool, elapsed_ms: float = 0.0) -> None:
    """Emit a structured fetch log line."""
    status = "CACHE" if cached else f"LIVE  {elapsed_ms:.0f}ms"
    logger.info(f"{source:<28} {key:<30} {status}")
