#!/usr/bin/env python3
"""
_http.py — Shared HTTP helpers for all data_sources modules.

Provides:
    get(url, *, headers=None, timeout=10, **kwargs) -> requests.Response
    get_cached_http(cache_key_str, url, *, headers=None, timeout=10, ttl=1800) -> dict
"""

import time
import requests

from data_sources._cache import cache_get, cache_set, log_fetch

_DEFAULT_HEADERS = {"User-Agent": "StockResearchAgent/1.0"}


def get(url: str, *, headers: dict = None, timeout: int = 10, **kwargs) -> requests.Response:
    """
    Thin wrapper around requests.get with default headers and timeout.
    Merges caller-supplied headers on top of the default User-Agent.
    Raises HTTPError on non-2xx status.
    """
    merged = {**_DEFAULT_HEADERS, **(headers or {})}
    r = requests.get(url, headers=merged, timeout=timeout, **kwargs)
    r.raise_for_status()
    return r


def get_cached_http(
    cache_key_str: str,
    url: str,
    *,
    source_label: str = "HTTP",
    headers: dict = None,
    timeout: int = 10,
    ttl: int = 1800,
    **kwargs,
) -> dict:
    """
    Fetch url with caching. Returns the parsed JSON as a dict.
    On cache hit, returns the cached dict without making a network call.
    On miss, fetches, parses JSON, stores result, and returns it.
    """
    hit = cache_get(cache_key_str, ttl=ttl)
    if hit:
        log_fetch(source_label, cache_key_str, cached=True)
        return hit

    t0 = time.time()
    merged = {**_DEFAULT_HEADERS, **(headers or {})}
    r = requests.get(url, headers=merged, timeout=timeout, **kwargs)
    r.raise_for_status()
    data = r.json()
    elapsed = round((time.time() - t0) * 1000)
    if isinstance(data, dict):
        data["_elapsed_ms"] = elapsed
    log_fetch(source_label, cache_key_str, cached=False, elapsed_ms=elapsed)
    cache_set(cache_key_str, data if isinstance(data, dict) else {"_data": data, "_elapsed_ms": elapsed})
    return data
