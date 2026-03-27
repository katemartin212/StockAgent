#!/usr/bin/env python3
"""
comps_data.py — Peer comparison (comps) table data via yfinance.

Fetches valuation metrics for a subject company and its peers in parallel.
Computes peer medians and premium/discount indicators.

Functions:
    fetch_comps(subject_ticker, peers, sector) → dict
"""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_sources._cache import cache_get, cache_set, cache_key, log_fetch, logger

SECTOR_ETF = {
    "Technology":             "XLK",
    "Communication Services": "XLC",
    "Healthcare":             "XLV",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Energy":                 "XLE",
    "Real Estate":            "XLRE",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Utilities":              "XLU",
}

# Columns used in the comps table
NUMERIC_COLS = [
    "market_cap_b",
    "revenue_growth_pct",
    "gross_margin_pct",
    "ev_revenue",
    "ev_ebitda",
    "forward_pe",
    "analyst_target",
]

# For these columns, a LOWER value is better (invert green/red tint logic)
LOWER_IS_BETTER = {"ev_revenue", "ev_ebitda", "forward_pe"}


def _safe(info: dict, key: str, divisor: float = 1.0,
          min_val: float | None = None, max_val: float | None = None) -> float | None:
    v = info.get(key)
    if v is None:
        return None
    try:
        f = float(v) / divisor
        if min_val is not None and f < min_val:
            return None
        if max_val is not None and f > max_val:
            return None
        return round(f, 2)
    except (TypeError, ValueError):
        return None


def _fetch_one(ticker: str) -> dict:
    """Fetch one ticker's comps metrics from yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return {
            "ticker":              ticker.upper(),
            "name":                (info.get("shortName") or info.get("longName") or ticker)[:26],
            "market_cap_b":        _safe(info, "marketCap", 1e9, 0),
            "revenue_growth_pct":  round(float(info["revenueGrowth"]) * 100, 1) if info.get("revenueGrowth") is not None else None,
            "gross_margin_pct":    round(float(info["grossMargins"]) * 100, 1) if info.get("grossMargins") is not None else None,
            "ev_revenue":          _safe(info, "enterpriseToRevenue", 1, 0),
            "ev_ebitda":           _safe(info, "enterpriseToEbitda", 1, 0, 500),
            "forward_pe":          _safe(info, "forwardPE", 1, 0, 500),
            "analyst_target":      _safe(info, "targetMeanPrice"),
            "consensus":           (info.get("recommendationKey") or "").replace("_", " ").title() or None,
        }
    except Exception as e:
        logger.warning(f"comps _fetch_one({ticker}): {e}")
        return {"ticker": ticker.upper(), "name": ticker, "error": str(e)}


def _median_row(rows: list[dict]) -> dict:
    result = {"ticker": "MEDIAN", "name": "Peer Median", "is_median": True}
    for col in NUMERIC_COLS:
        vals = [r[col] for r in rows if r.get(col) is not None]
        result[col] = round(statistics.median(vals), 2) if vals else None
    return result


def fetch_comps(subject_ticker: str,
                peers: list[str],
                sector: str | None = None) -> dict:
    """
    Fetch comps data for subject + peers + sector ETF benchmark.
    Caches result for 60 minutes.

    Returns dict with:
        rows          list of metric dicts (subject first, then peers, then ETF)
        median        peer median row (excludes subject and ETF)
        etf_ticker    which ETF was used
        col_ranges    {col: {min, max}} for color-coding
    """
    subject_ticker = subject_ticker.upper()
    peers = [p.upper() for p in peers[:6] if p.upper() != subject_ticker]
    etf   = SECTOR_ETF.get(sector or "", "SPY")

    all_tickers = [subject_ticker] + peers
    if etf not in all_tickers:
        all_tickers.append(etf)

    ck  = cache_key("comps", subject_ticker, *sorted(peers))
    hit = cache_get(ck, ttl=3600)
    if hit:
        log_fetch("Comps", subject_ticker, cached=True)
        return hit

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=min(len(all_tickers), 8)) as pool:
        futs = {pool.submit(_fetch_one, t): t for t in all_tickers}
        results_map: dict[str, dict] = {}
        for fut in as_completed(futs, timeout=25):
            t = futs[fut]
            try:
                results_map[t] = fut.result()
            except Exception as e:
                results_map[t] = {"ticker": t, "name": t, "error": str(e)}

    # Build ordered rows: subject → peers → ETF
    rows = []
    for t in all_tickers:
        r = results_map.get(t, {"ticker": t, "name": t, "error": "timeout"})
        r["is_subject"] = (t == subject_ticker)
        r["is_etf"]     = (t == etf)
        rows.append(r)

    # Peer median (exclude subject, ETF, and errored rows)
    peer_rows = [
        r for r in rows
        if not r.get("is_subject") and not r.get("is_etf") and not r.get("error")
    ]
    median = _median_row(peer_rows) if peer_rows else None

    # Premium / discount vs median for the subject row
    subject_row = rows[0]
    if median:
        for col in ("ev_revenue", "ev_ebitda", "forward_pe"):
            sv = subject_row.get(col)
            mv = median.get(col)
            if sv is not None and mv and mv > 0:
                subject_row[f"{col}_vs_median_pct"] = round((sv - mv) / mv * 100, 1)

    # Column min/max for heat-map coloring (peer rows only, so subject isn't trivially extreme)
    col_ranges: dict[str, dict] = {}
    range_rows = [r for r in peer_rows] + ([median] if median else [])
    for col in NUMERIC_COLS:
        vals = [r[col] for r in range_rows if r.get(col) is not None]
        if len(vals) >= 2:
            col_ranges[col] = {"min": min(vals), "max": max(vals)}

    out = {
        "subject_ticker": subject_ticker,
        "rows":           rows,
        "median":         median,
        "etf_ticker":     etf,
        "col_ranges":     col_ranges,
        "lower_is_better": list(LOWER_IS_BETTER),
        "_elapsed_ms":    round((time.time() - t0) * 1000),
    }
    log_fetch("Comps", subject_ticker, cached=False, elapsed_ms=out["_elapsed_ms"])
    cache_set(ck, out)
    return out
