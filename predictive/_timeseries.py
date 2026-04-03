#!/usr/bin/env python3
"""
_timeseries.py — Weekly price alignment and statistical helpers.

Functions:
    align_weekly(symbol, period)   Fetch + TZ-normalize + resample to W-FRI
    robust_sigma(log_changes)      MAD-based annualized σ (√52 scaled)
    zscore_log(series, value)      Z-score in log-change space
    decay_weights(n, lam)          Exponential decay weight vector (newest = highest)
"""

import numpy as np
import pandas as pd


def align_weekly(symbol: str, period: str = "2y") -> pd.Series:
    """
    Fetch weekly adjusted close for *symbol*, strip timezone, resample to W-FRI.
    Returns an empty Series on any error.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, interval="1wk", auto_adjust=True)
        if hist.empty:
            return pd.Series(dtype=float, name=symbol)
        s = hist["Close"].rename(symbol)
        if s.index.tz is not None:
            s = s.tz_convert("UTC").tz_localize(None)
        return s.resample("W-FRI").last()
    except Exception:
        return pd.Series(dtype=float, name=symbol)


def robust_sigma(log_changes: np.ndarray) -> float:
    """
    MAD-based weekly σ scaled to annual (√52).
    MAD is divided by 0.6745 to convert to a consistent σ estimate under normality.
    Returns 0.0 if fewer than 4 observations.
    """
    arr = np.asarray(log_changes, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return 0.0
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    weekly_sigma = mad / 0.6745
    return float(weekly_sigma * np.sqrt(52))


def zscore_log(series: pd.Series, current_val: float) -> float:
    """
    Compute z-score of *current_val* relative to the log-change distribution of *series*.
    Uses robust_sigma (MAD-based) to avoid outlier inflation.
    Returns 0.0 if series is too short or sigma is zero.
    """
    s = series.dropna()
    if len(s) < 8:
        return 0.0
    log_chg = np.log(s / s.shift(1)).dropna().values
    sigma = robust_sigma(log_chg)
    if sigma == 0.0:
        return 0.0
    median_val = float(np.median(s))
    if median_val <= 0:
        return 0.0
    log_z = np.log(max(current_val, 1e-9) / median_val) / (sigma / np.sqrt(52))
    return float(np.clip(log_z, -5.0, 5.0))


def decay_weights(n: int, lam: float = 0.98) -> np.ndarray:
    """
    Exponential decay weight vector of length *n*.
    Newest observation (index n-1) receives weight 1.0; oldest receives lam^(n-1).
    Weights are returned unnormalized — pass to _ridge.weighted_ridge as-is.
    """
    return lam ** np.arange(n - 1, -1, -1)
