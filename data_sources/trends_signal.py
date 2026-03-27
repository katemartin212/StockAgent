#!/usr/bin/env python3
"""
trends_signal.py — Google Trends search interest via pytrends.

Requires: pip install pytrends

Functions:
    get_search_interest(ticker, company_name=None) → dict
    get_comparative_interest(tickers)              → dict
"""

import time
from datetime import datetime, timedelta

from data_sources._cache import cache_get, cache_set, cache_key, log_fetch, logger

try:
    from pytrends.request import TrendReq
    _PYTRENDS_AVAILABLE = True
except ImportError:
    _PYTRENDS_AVAILABLE = False


def _not_installed() -> dict:
    return {
        "error":       "pytrends not installed. Run: pip install pytrends",
        "data_source": "Google Trends",
    }


def get_search_interest(ticker: str, company_name: str | None = None) -> dict:
    """
    Fetch 12-month Google Trends interest for a ticker (and optionally the
    company name). Returns current interest vs 3-month average, the peak date
    and value, a near-peak signal, and related queries.
    """
    if not _PYTRENDS_AVAILABLE:
        return _not_installed()

    ck = cache_key("trends_interest", ticker)
    hit = cache_get(ck)
    if hit:
        log_fetch("GoogleTrends", ticker, cached=True)
        return hit

    t0 = time.time()

    try:
        # Primary search term is "$TICKER"; add company name as second term if given
        kw_ticker  = f"${ticker.upper()}"
        keywords   = [kw_ticker]
        if company_name:
            keywords.append(company_name[:60])   # pytrends limit

        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=0)
        pytrends.build_payload(keywords[:1], timeframe="today 12-m", geo="")
        time.sleep(1.0)   # Google rate-limiting buffer

        interest_df = pytrends.interest_over_time()

        if interest_df.empty:
            out = {
                "ticker":          ticker.upper(),
                "search_term":     kw_ticker,
                "data_available":  False,
                "signal_note":     "No Google Trends data found (low search volume).",
                "data_source":     "Google Trends",
                "_elapsed_ms":     round((time.time() - t0) * 1000),
            }
            cache_set(ck, out)
            return out

        col = kw_ticker
        if col not in interest_df.columns:
            col = interest_df.columns[0]

        series = interest_df[col].dropna()
        values = series.tolist()
        dates  = [str(d.date()) for d in series.index]

        current_val    = values[-1]
        avg_3mo        = round(sum(values[-13:]) / len(values[-13:]), 1) if len(values) >= 13 else round(sum(values) / len(values), 1)
        avg_12mo       = round(sum(values) / len(values), 1)
        peak_val       = max(values)
        peak_idx       = values.index(peak_val)
        peak_date      = dates[peak_idx] if peak_idx < len(dates) else None
        near_peak      = current_val >= peak_val * 0.85

        trend_vs_avg   = "above_avg" if current_val > avg_3mo * 1.1 else ("below_avg" if current_val < avg_3mo * 0.9 else "at_avg")

        # Related queries
        time.sleep(0.5)
        try:
            related = pytrends.related_queries()
            top_related = []
            kw_data = related.get(col) or related.get(list(related.keys())[0], {})
            top_df  = kw_data.get("top")
            if top_df is not None and not top_df.empty:
                top_related = top_df["query"].head(5).tolist()
        except Exception:
            top_related = []

        out = {
            "ticker":          ticker.upper(),
            "search_term":     col,
            "current_interest": current_val,     # 0–100 relative scale
            "avg_3mo":         avg_3mo,
            "avg_12mo":        avg_12mo,
            "peak_value":      peak_val,
            "peak_date":       peak_date,
            "near_peak":       near_peak,
            "trend_vs_avg":    trend_vs_avg,
            "related_queries": top_related,
            "data_available":  True,
            "signal_note": (
                f"Search interest {current_val}/100 — "
                f"{'near peak' if near_peak else trend_vs_avg.replace('_', ' ')}. "
                f"12mo avg: {avg_12mo}."
            ),
            "data_source":     "Google Trends (pytrends)",
            "_elapsed_ms":     round((time.time() - t0) * 1000),
        }

        log_fetch("GoogleTrends", ticker, cached=False, elapsed_ms=out["_elapsed_ms"])
        cache_set(ck, out)
        return out

    except Exception as e:
        out = {"error": str(e), "ticker": ticker, "data_source": "Google Trends"}
        logger.error(f"get_search_interest({ticker}): {e}")
        return out


def get_comparative_interest(tickers: list[str]) -> dict:
    """
    Compare relative Google Trends search interest for up to 5 tickers.
    Returns a ranked list with relative interest values (0–100 scale
    normalized to the highest ticker in the group).
    """
    if not _PYTRENDS_AVAILABLE:
        return _not_installed()

    if not tickers:
        return {"error": "No tickers provided", "data_source": "Google Trends"}

    tickers = [t.upper() for t in tickers[:5]]   # max 5
    ck = cache_key("trends_compare", *sorted(tickers))
    hit = cache_get(ck)
    if hit:
        log_fetch("GoogleTrends", f"compare:{','.join(tickers)}", cached=True)
        return hit

    t0 = time.time()

    try:
        keywords = [f"${t}" for t in tickers]
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=0)
        pytrends.build_payload(keywords, timeframe="today 3-m", geo="")
        time.sleep(0.5)

        df = pytrends.interest_over_time()

        if df.empty:
            out = {
                "tickers":         tickers,
                "data_available":  False,
                "signal_note":     "No comparative Google Trends data found.",
                "data_source":     "Google Trends",
                "_elapsed_ms":     round((time.time() - t0) * 1000),
            }
            cache_set(ck, out)
            return out

        rankings = []
        for kw, ticker in zip(keywords, tickers):
            if kw in df.columns:
                avg = round(df[kw].mean(), 1)
                rankings.append({"ticker": ticker, "avg_interest": avg, "search_term": kw})

        rankings.sort(key=lambda x: x["avg_interest"], reverse=True)

        out = {
            "tickers":        tickers,
            "rankings":       rankings,
            "data_available": True,
            "signal_note":    f"Search interest leader: {rankings[0]['ticker']} ({rankings[0]['avg_interest']}/100)." if rankings else "No data.",
            "data_source":    "Google Trends (pytrends)",
            "_elapsed_ms":    round((time.time() - t0) * 1000),
        }

        log_fetch("GoogleTrends", f"compare:{','.join(tickers)}", cached=False, elapsed_ms=out["_elapsed_ms"])
        cache_set(ck, out)
        return out

    except Exception as e:
        out = {"error": str(e), "tickers": tickers, "data_source": "Google Trends"}
        logger.error(f"get_comparative_interest({tickers}): {e}")
        return out
