#!/usr/bin/env python3
"""
tools_base.py — Shared utilities for all tools_*.py sector modules.
"""

import json
import yfinance as yf

# Minimal info fields needed across sector tools
_BASE_FIELDS = {
    "longName", "shortName", "sector", "industry",
    "currentPrice", "regularMarketPrice",
    "marketCap", "enterpriseValue",
    "totalRevenue", "grossProfits", "freeCashflow",
    "grossMargins", "operatingMargins", "profitMargins",
    "revenueGrowth", "earningsGrowth",
    "sharesOutstanding", "floatShares",
    "totalDebt", "totalCash",
    "beta", "debtToEquity", "dividendYield",
    "fullTimeEmployees", "longBusinessSummary",
    "returnOnEquity", "returnOnAssets",
}


def fetch_info(ticker_or_obj, extra_fields=None) -> dict:
    """Return filtered info dict for a ticker string or yf.Ticker object."""
    if isinstance(ticker_or_obj, str):
        t = yf.Ticker(ticker_or_obj.upper().strip())
    else:
        t = ticker_or_obj
    return {k: v for k, v in t.info.items() if k in (_BASE_FIELDS | (extra_fields or set()))}


def _find_series(df, candidates: list):
    """
    Search df.index for each candidate row name in order.
    Returns the first non-empty Series found, or None.
    """
    if df is None:
        return None
    try:
        for name in candidates:
            if name in df.index:
                s = df.loc[name].dropna()
                if len(s) > 0:
                    return s
    except Exception:
        return None
    return None


def safe_ratio(num, denom, decimals: int = 2, pct: bool = False, default=None):
    """
    Safely compute num/denom. Multiplies by 100 if pct=True.
    Returns default on ZeroDivisionError or TypeError.
    """
    try:
        if not denom:
            return default
        result = num / denom
        return round(result * (100 if pct else 1), decimals)
    except (TypeError, ZeroDivisionError):
        return default


def tool_result(data: dict = None, error: str = None) -> str:
    """Serialize tool output to JSON string. Pass error= for error responses."""
    if error is not None:
        return json.dumps({"error": error})
    return json.dumps(data)


def _tool_schema(name: str, description: str, example: str = "AAPL") -> dict:
    """Generate a standard single-ticker tool definition dict."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": f"Ticker symbol, e.g. '{example}'"
                }
            },
            "required": ["ticker"]
        }
    }
