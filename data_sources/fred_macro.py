#!/usr/bin/env python3
"""
fred_macro.py — Macro environment data from multiple free sources.

Primary sources (no API key required):
  yfinance  — ^TNX (10Y Treasury), ^IRX (3M Treasury), CL=F (WTI crude)
  BLS API   — CPI (CUUR0000SA0), Unemployment (LNS14000000)
  FRED CSV  — attempted as fallback when yfinance/BLS is insufficient
              (fred.stlouisfed.org/graph/fredgraph.csv)

Functions:
    get_fred_macro() → dict
"""

import os
import time
import json
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_sources._cache import cache_get, cache_set, cache_key, log_fetch, logger

HEADERS = {"User-Agent": "StockResearchAgent/1.0"}

# ── yfinance Treasury / commodity tickers ─────────────────────────────────────
YF_TICKERS = {
    "^TNX":  ("DGS10",      "10Y Treasury"),
    "^IRX":  ("DGS3M",      "3M Treasury"),
    "CL=F":  ("DCOILWTICO", "WTI Crude $/bbl"),
}

# ── BLS series IDs (free public API, no key needed) ─────────────────────────
BLS_SERIES = {
    "CUUR0000SA0": ("CPIAUCSL", "CPI (level)"),
    "LNS14000000": ("UNRATE",   "Unemployment %"),
}
BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

# ── FRED CSV (fallback, often unreliable) ────────────────────────────────────
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES_FALLBACK = {
    "FEDFUNDS":   "Fed Funds %",
    "DGS2":       "2Y Treasury",
}


def _entry(value: float, prev: float, label: str) -> dict:
    trend = ("rising"  if value > prev * 1.005 else
             "falling" if value < prev * 0.995 else "stable")
    return {"value": round(value, 3), "prev": round(prev, 3),
            "trend": trend, "label": label}


def _fetch_yf() -> dict[str, dict]:
    """Fetch Treasury yields and WTI crude from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return {}

    results = {}
    try:
        tickers = yf.download(
            list(YF_TICKERS.keys()),
            period="5d", interval="1d",
            progress=False, auto_adjust=True,
        )
        closes = tickers.get("Close", tickers) if hasattr(tickers, "get") else tickers
        for sym, (sid, label) in YF_TICKERS.items():
            try:
                col = closes[sym] if sym in closes.columns else closes
                col = col.dropna()
                if len(col) >= 2:
                    val  = float(col.iloc[-1])
                    prev = float(col.iloc[-2])
                    # yfinance already returns yields in % (e.g. 4.391 = 4.391%)
                    results[sid] = _entry(val, prev, label)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"yfinance macro fetch: {e}")
    return results


def _fetch_bls() -> dict[str, dict]:
    """Fetch CPI and Unemployment from BLS public API (no key needed)."""
    year_now = datetime.now().year
    payload = json.dumps({
        "seriesid":  list(BLS_SERIES.keys()),
        "startyear": str(year_now - 1),
        "endyear":   str(year_now),
    })
    try:
        r = requests.post(BLS_URL, data=payload,
                          headers={"Content-Type": "application/json"},
                          timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "REQUEST_SUCCEEDED":
            return {}

        results = {}
        for series_obj in data.get("Results", {}).get("series", []):
            bls_id = series_obj.get("seriesID", "")
            sid, label = BLS_SERIES.get(bls_id, (None, None))
            if not sid:
                continue
            obs = sorted(
                series_obj.get("data", []),
                key=lambda x: (x.get("year", ""), x.get("period", "")),
            )
            valid = [float(o["value"]) for o in obs
                     if o.get("value") not in ("", "-")]
            if len(valid) >= 2:
                entry = _entry(valid[-1], valid[-2], label)
                if sid == "CPIAUCSL" and len(valid) >= 13:
                    yoy = round((valid[-1] - valid[-13]) / valid[-13] * 100, 2)
                    entry["yoy_pct"] = yoy
                results[sid] = entry
        return results
    except Exception as e:
        logger.warning(f"BLS macro fetch: {e}")
        return {}


def _fetch_fred_csv(series_id: str, label: str) -> tuple[str, dict | None]:
    """Attempt FRED CSV for a single series (short timeout, best-effort)."""
    try:
        url = f"{FRED_CSV_BASE}?id={series_id}"
        r = requests.get(url, timeout=6, headers=HEADERS)
        r.raise_for_status()
        rows = []
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 2 or parts[1].strip() in ("", "."):
                continue
            try:
                rows.append(float(parts[1].strip()))
            except ValueError:
                pass
        if len(rows) >= 2:
            return series_id, _entry(rows[-1], rows[-2], label)
    except Exception:
        pass
    return series_id, None


def get_fred_macro() -> dict:
    """
    Fetch macro environment data from multiple free sources in parallel.
    Sources: yfinance (Treasuries, WTI), BLS (CPI, Unemployment),
    FRED CSV (Fed Funds, 2Y Treasury — best-effort).
    """
    ck = cache_key("fred_macro", "all")
    cached = cache_get(ck, ttl=3600)
    if cached:
        log_fetch("FRED/macro", "macro_all", cached=True)
        return cached

    t0 = time.time()
    result_series: dict[str, dict] = {}
    errors: list[str] = []

    # Run all sources in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_yf  = pool.submit(_fetch_yf)
        fut_bls = pool.submit(_fetch_bls)
        fred_futs = {
            pool.submit(_fetch_fred_csv, sid, label): sid
            for sid, label in FRED_SERIES_FALLBACK.items()
        }

        try:
            yf_data = fut_yf.result(timeout=10)
            result_series.update(yf_data)
        except Exception as e:
            errors.append(f"yfinance: {e}")

        try:
            bls_data = fut_bls.result(timeout=12)
            result_series.update(bls_data)
        except Exception as e:
            errors.append(f"BLS: {e}")

        for fut in as_completed(fred_futs, timeout=8):
            try:
                sid, entry = fut.result()
                if entry:
                    result_series[sid] = entry
            except Exception as e:
                errors.append(f"FRED CSV: {e}")

    # ── Derived indicators ────────────────────────────────────────────────────

    # 2s10s spread (use 10Y and 3M as proxy if 2Y unavailable)
    spread_2s10s = None
    dgs10 = result_series.get("DGS10")
    dgs2  = result_series.get("DGS2") or result_series.get("DGS3M")
    if dgs10 and dgs2:
        spread = round(dgs10["value"] - dgs2["value"], 3)
        spread_2s10s = {
            "value":    spread,
            "inverted": spread < 0,
            "label":    "10Y–2Y Spread" if "DGS2" in result_series else "10Y–3M Spread",
        }

    cpi_yoy   = result_series.get("CPIAUCSL", {}).get("yoy_pct")
    ff_trend  = result_series.get("FEDFUNDS",  {}).get("trend", "stable")
    dgs10_trend = result_series.get("DGS10",   {}).get("trend", "stable")

    # Regime: use FEDFUNDS trend if available, else infer from 10Y direction
    regime = ("Easing"     if ff_trend == "falling" else
              "Tightening" if ff_trend == "rising"  else
              "Easing"     if dgs10_trend == "falling" else
              "Tightening" if dgs10_trend == "rising"  else "Stable")

    # Plain English
    sentences = []
    ff_val = result_series.get("FEDFUNDS", {}).get("value")
    dgs10_val = result_series.get("DGS10", {}).get("value")
    if ff_val:
        sentences.append(f"Fed Funds at {ff_val}%.")
    elif dgs10_val:
        sentences.append(f"10Y Treasury at {dgs10_val}% — {dgs10_trend}.")

    if regime == "Easing":
        sentences.append("Rate environment easing — multiple expansion tailwind for growth equities.")
    elif regime == "Tightening":
        sentences.append("Rate environment tightening — valuation headwind for high-multiple stocks.")
    else:
        sentences.append("Rates stable — no near-term repricing catalyst.")

    if spread_2s10s and spread_2s10s["inverted"]:
        sentences.append("Yield curve inverted — historically a recession leading indicator.")
    elif spread_2s10s and spread_2s10s["value"] > 0.3:
        sentences.append("Yield curve re-steepening — late-cycle signal, positive for banks.")

    if cpi_yoy is not None:
        if cpi_yoy < 2.5:
            sentences.append(f"CPI YoY {cpi_yoy}% — near Fed target, supports rate cuts.")
        elif cpi_yoy > 4.0:
            sentences.append(f"CPI YoY {cpi_yoy}% — elevated inflation constrains policy easing.")
        else:
            sentences.append(f"CPI YoY {cpi_yoy}% — moderating but above target.")

    sources_used = []
    if any(k in result_series for k in ("DGS10", "DGS3M", "DCOILWTICO")):
        sources_used.append("yfinance")
    if any(k in result_series for k in ("CPIAUCSL", "UNRATE")):
        sources_used.append("BLS")
    if any(k in result_series for k in ("FEDFUNDS", "DGS2")):
        sources_used.append("FRED CSV")

    out = {
        "as_of":         datetime.now().strftime("%Y-%m-%d"),
        "macro_regime":  regime,
        "series":        result_series,
        "spread_2s10s":  spread_2s10s,
        "cpi_yoy_pct":   cpi_yoy,
        "plain_english": " ".join(sentences) if sentences else "Macro data unavailable.",
        "errors":        errors if errors else None,
        "data_source":   f"Macro: {', '.join(sources_used) or 'unavailable'}",
        "_elapsed_ms":   round((time.time() - t0) * 1000),
    }

    log_fetch("FRED/macro", "macro_all", cached=False, elapsed_ms=out["_elapsed_ms"])
    cache_set(ck, out)
    return out
