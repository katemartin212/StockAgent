#!/usr/bin/env python3
"""
master_signal.py — Parallel data orchestration and composite signal scoring.

Runs all data sources concurrently via ThreadPoolExecutor and synthesizes
a composite JSON with fundamental, retail sentiment, macro, and insider scores.

Functions:
    get_master_analysis(ticker, sector=None) → dict
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

logger = logging.getLogger("stock_agent")

# ── Data source imports (all fail gracefully if module unavailable) ──────────

def _safe_import(module_path: str, fn_name: str):
    """Return callable or a lambda that returns an error dict."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, fn_name)
    except Exception as e:
        logger.warning(f"Could not import {module_path}.{fn_name}: {e}")
        return lambda *a, **kw: {"error": f"{module_path} unavailable: {e}", "data_source": module_path}


_get_edgar_financials      = _safe_import("data_sources.sec_edgar",           "get_edgar_financials")
_get_edgar_form4           = _safe_import("data_sources.sec_edgar",           "get_edgar_form4")
_get_edgar_filings         = _safe_import("data_sources.sec_edgar",           "get_edgar_filings")
_get_fred_macro            = _safe_import("data_sources.fred_macro",          "get_fred_macro")
_get_open_insider          = _safe_import("data_sources.open_insider",        "get_open_insider")
_get_reddit_sentiment      = _safe_import("data_sources.reddit_sentiment",    "get_reddit_sentiment")
_get_stocktwits_sentiment  = _safe_import("data_sources.stocktwits_sentiment","get_stocktwits_sentiment")
_get_search_interest       = _safe_import("data_sources.trends_signal",       "get_search_interest")

# Per-source timeout (seconds)
SOURCE_TIMEOUT = 20


# ── Scoring helpers ──────────────────────────────────────────────────────────

def _fundamental_score(edgar: dict) -> float | None:
    """
    Composite fundamental score 0-100.
    Rule of 40 (30%), gross-margin trend (20%), FCF yield proxy (25%),
    dilution (15%), revenue growth (10%).
    Returns None if insufficient data.
    """
    if "error" in edgar or not edgar.get("quarterly"):
        return None

    q = edgar.get("quarterly", [])
    if len(q) < 4:
        return None

    scores = []

    # Revenue growth (YoY, most recent quarter vs 4 quarters ago)
    try:
        rev_now  = q[-1].get("revenue") or 0
        rev_year = q[-5].get("revenue") if len(q) >= 5 else (q[0].get("revenue") or 0)
        if rev_year and rev_year > 0:
            yoy_rev_growth = (rev_now - rev_year) / rev_year * 100
            # Score: 0→40pts, 20%→65pts, 40%→90pts, saturates at 100
            rev_score = min(100, max(0, 50 + yoy_rev_growth * 1.25))
            scores.append(("revenue_growth", rev_score, 0.10))
    except Exception:
        pass

    # Gross margin trend (recent 4q average vs prior 4q average)
    try:
        gms = [r.get("gross_margin_pct") for r in q if r.get("gross_margin_pct") is not None]
        if len(gms) >= 6:
            recent_gm = sum(gms[-4:]) / 4
            prior_gm  = sum(gms[-8:-4]) / 4 if len(gms) >= 8 else sum(gms[:-4]) / len(gms[:-4])
            delta_gm  = recent_gm - prior_gm
            gm_score  = min(100, max(0, 50 + delta_gm * 5 + (recent_gm - 40) * 0.5))
            scores.append(("gm_trend", gm_score, 0.20))
    except Exception:
        pass

    # Rule of 40 (revenue_growth_yoy + fcf_margin)
    try:
        fcf_now  = q[-1].get("fcf") or 0
        if rev_now and rev_now > 0 and 'yoy_rev_growth' in dir():
            fcf_margin = (fcf_now / rev_now) * 100
            rule40     = yoy_rev_growth + fcf_margin
            r40_score  = min(100, max(0, 50 + (rule40 - 30) * 1.5))
            scores.append(("rule_of_40", r40_score, 0.30))
    except Exception:
        pass

    # Dilution (shares YoY change — lower dilution = better)
    try:
        shares_now  = q[-1].get("shares_outstanding") or 0
        shares_year = q[-5].get("shares_outstanding") if len(q) >= 5 else (q[0].get("shares_outstanding") or 0)
        if shares_year and shares_year > 0 and shares_now > 0:
            dilution_pct = (shares_now - shares_year) / shares_year * 100
            dil_score    = min(100, max(0, 80 - dilution_pct * 10))
            scores.append(("dilution", dil_score, 0.15))
    except Exception:
        pass

    # FCF positivity / magnitude
    try:
        fcf_vals = [r.get("fcf") or 0 for r in q[-4:]]
        positive_quarters = sum(1 for f in fcf_vals if f > 0)
        fcf_score = (positive_quarters / 4) * 100
        scores.append(("fcf_quality", fcf_score, 0.25))
    except Exception:
        pass

    if not scores:
        return None

    total_weight = sum(w for _, _, w in scores)
    composite    = sum(s * w for _, s, w in scores) / total_weight
    return round(composite, 1)


def _insider_score(form4: dict, open_insider: dict) -> float:
    """
    Insider conviction score 0-100.
    Combines weighted signal from EDGAR Form 4 + OpenInsider cluster signal.
    """
    score = 50.0   # neutral baseline

    # EDGAR Form 4 weighted signal
    ws = (form4 or {}).get("weighted_signal")
    if ws is not None:
        # weighted_signal: positive = buying, negative = selling
        # Map to 0-100: 0→25, neutral→50, strong buy→90
        score += min(40, max(-40, ws * 8))

    # OpenInsider cluster buy
    oi_signal = (open_insider or {}).get("signal", "")
    if oi_signal == "cluster_buy":
        score = min(100, score + 20)
    elif oi_signal == "net_buying":
        score = min(100, score + 10)
    elif oi_signal == "net_selling":
        score = max(0, score - 10)

    return round(min(100, max(0, score)), 1)


def _retail_sentiment_score(reddit: dict, stocktwits: dict, trends: dict) -> float:
    """
    Retail sentiment score 0-100.
    Reddit (30%), StockTwits (30%), Google Trends (40%).
    """
    parts = []

    # Reddit: sentiment_score is -1 to +1; map to 0-100
    rs = (reddit or {}).get("sentiment_score")
    if rs is not None and "error" not in (reddit or {}):
        reddit_score = round((rs + 1) / 2 * 100, 1)
        parts.append((reddit_score, 0.30))

    # StockTwits: sentiment_score already 0-100
    ss = (stocktwits or {}).get("sentiment_score")
    if ss is not None and "error" not in (stocktwits or {}):
        parts.append((ss, 0.30))

    # Google Trends: current_interest / 100 * 100; near_peak adds 10 bonus
    ti = (trends or {}).get("current_interest")
    if ti is not None and "error" not in (trends or {}):
        t_score = min(100, ti + (10 if (trends or {}).get("near_peak") else 0))
        parts.append((t_score, 0.40))

    if not parts:
        return 50.0

    total_w = sum(w for _, w in parts)
    return round(sum(s * w for s, w in parts) / total_w, 1)


def _macro_score(macro: dict, sector: str | None) -> float:
    """
    Macro tailwind score 0-100.
    50 = neutral. Adjusts based on regime and yield curve.
    """
    if not macro or "error" in macro:
        return 50.0

    score = 50.0
    regime = macro.get("macro_regime", "Stable")

    if regime == "Easing":
        score += 15
    elif regime == "Tightening":
        score -= 15

    spread_obj = macro.get("spread_2s10s") or {}
    if spread_obj.get("inverted"):
        score -= 10
    elif (spread_obj.get("value") or 0) > 0.3:
        score += 5

    cpi = macro.get("cpi_yoy_pct")
    if cpi is not None:
        if cpi < 2.5:
            score += 5
        elif cpi > 4.5:
            score -= 10

    # Sector-specific macro tilts
    if sector:
        s = sector.lower()
        if "real estate" in s or "reit" in s:
            # REITs very rate sensitive
            score += -20 if regime == "Tightening" else 10 if regime == "Easing" else 0
        elif "financial" in s or "bank" in s:
            # Banks benefit from rising rates (NIM)
            score += 10 if regime == "Tightening" else -5 if regime == "Easing" else 0
        elif "technology" in s:
            # Tech benefits from easing (multiple expansion)
            score += 10 if regime == "Easing" else -10 if regime == "Tightening" else 0

    return round(min(100, max(0, score)), 1)


def _divergence_score(fundamental: float | None, retail: float) -> float:
    """
    Divergence score 0-100: measures gap between fundamental quality and retail sentiment.
    High score = strong fundamentals + low retail attention (potential opportunity).
    Low score  = weak fundamentals + high retail excitement (potential danger).
    """
    if fundamental is None:
        return 50.0
    # +50 offset so neutral=50; divergence is fundamental strength minus retail hype
    divergence = (fundamental - retail) / 2 + 50
    return round(min(100, max(0, divergence)), 1)


def _recommended_action(fund: float | None, retail: float, insider: float, macro: float) -> str:
    weights = [(fund or 50, 0.40), (insider, 0.25), (retail, 0.20), (macro, 0.15)]
    composite = sum(s * w for s, w in weights)
    if composite >= 70:
        return "BUY / ACCUMULATE"
    if composite >= 55:
        return "OVERWEIGHT — monitor"
    if composite >= 45:
        return "HOLD / NEUTRAL"
    if composite >= 30:
        return "UNDERWEIGHT — reduce"
    return "SELL / AVOID"


# ── Main orchestration ───────────────────────────────────────────────────────

def get_master_analysis(ticker: str, sector: str | None = None) -> dict:
    """
    Run all data sources in parallel and return a composite analysis dict.

    Returns:
        fundamental_score      float 0-100
        insider_score          float 0-100
        retail_sentiment_score float 0-100
        macro_score            float 0-100
        divergence_score       float 0-100 (fundamental vs retail gap)
        composite_score        float 0-100
        recommended_action     str
        behavioral_signal      str  (from scores)
        dominant_retail_narrative str
        top_claims             list[str]
        key_risks              list[str]
        data_sources_used      list[str]
        data_freshness         dict
        raw                    dict (all source responses, for agent tool use)
    """
    t0 = time.time()
    ticker = ticker.upper()

    # Define tasks: (name, callable, args)
    tasks = {
        "edgar_financials": (_get_edgar_financials, (ticker,)),
        "edgar_form4":      (_get_edgar_form4,      (ticker,)),
        "edgar_filings":    (_get_edgar_filings,    (ticker,)),
        "fred_macro":       (_get_fred_macro,        ()),
        "open_insider":     (_get_open_insider,      (ticker,)),
        "reddit":           (_get_reddit_sentiment,  (ticker,)),
        "stocktwits":       (_get_stocktwits_sentiment, (ticker,)),
        "trends":           (_get_search_interest,   (ticker,)),
    }

    results = {}
    sources_used = []
    failed_sources = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(fn, *args): name
            for name, (fn, args) in tasks.items()
        }

        # Collect results as they complete; if the overall budget expires,
        # salvage whatever finished and mark the rest as timed-out.
        try:
            completed_iter = as_completed(future_map, timeout=SOURCE_TIMEOUT + 15)
            for future in completed_iter:
                name = future_map[future]
                try:
                    res = future.result(timeout=SOURCE_TIMEOUT)
                    results[name] = res
                    if "error" not in res:
                        sources_used.append(res.get("data_source", name))
                    else:
                        failed_sources.append(name)
                        logger.warning(f"get_master_analysis: {name} error: {res['error']}")
                except Exception as e:
                    results[name] = {"error": str(e)}
                    failed_sources.append(name)
                    logger.warning(f"get_master_analysis: {name} raised {e}")
        except TimeoutError:
            # Budget exhausted — mark any futures that never completed
            for future, name in future_map.items():
                if name not in results:
                    results[name] = {"error": f"{name} timed out"}
                    failed_sources.append(name)
                    logger.warning(f"get_master_analysis: {name} timed out (overall budget)")

    # Warn if 3+ sources failed
    if len(failed_sources) >= 3:
        logger.warning(f"get_master_analysis({ticker}): {len(failed_sources)} sources failed — {failed_sources}")

    # Compute scores
    fund_score    = _fundamental_score(results.get("edgar_financials", {}))
    insider_score = _insider_score(results.get("edgar_form4", {}), results.get("open_insider", {}))
    retail_score  = _retail_sentiment_score(
        results.get("reddit", {}),
        results.get("stocktwits", {}),
        results.get("trends", {}),
    )
    macro_score   = _macro_score(results.get("fred_macro", {}), sector)
    div_score     = _divergence_score(fund_score, retail_score)

    weights = [(fund_score or 50, 0.40), (insider_score, 0.25), (retail_score, 0.20), (macro_score, 0.15)]
    composite_score = round(sum(s * w for s, w in weights), 1)

    # Behavioral signal
    if div_score >= 65 and (fund_score or 50) >= 60:
        behavioral_signal = "UNDERVALUED — strong fundamentals, low retail attention"
    elif div_score <= 35 and retail_score >= 65:
        behavioral_signal = "OVERHYPED — weak fundamentals, high retail excitement"
    elif insider_score >= 75:
        behavioral_signal = "INSIDER CONVICTION — cluster buying or heavy open-market purchases"
    elif insider_score <= 30:
        behavioral_signal = "INSIDER CAUTION — notable selling pressure"
    else:
        behavioral_signal = "BALANCED — no extreme divergence detected"

    # Dominant retail narrative (simple synthesis from available signals)
    reddit_sentiment  = results.get("reddit", {}).get("overall_sentiment", "neutral")
    st_sentiment      = results.get("stocktwits", {}).get("overall_sentiment", "neutral")
    trends_note       = results.get("trends", {}).get("signal_note", "")
    dominant_narrative = (
        f"Reddit: {reddit_sentiment}. StockTwits: {st_sentiment}. "
        f"Search interest: {trends_note}"
    )

    # Key risks from data signals
    key_risks = []
    if insider_score < 35:
        key_risks.append("Significant insider selling — management may see headwinds")
    if macro_score < 35:
        key_risks.append("Macro headwinds — rising rates or tightening environment")
    if retail_score >= 75 and (fund_score or 50) < 45:
        key_risks.append("Retail over-enthusiasm without fundamental support")
    oi_signal = results.get("open_insider", {}).get("signal", "")
    if oi_signal == "net_selling":
        key_risks.append("Net insider selling on OpenInsider over last 90 days")
    if not key_risks:
        key_risks.append("No major risk signals detected from available data sources")

    # Data freshness
    data_freshness = {
        name: res.get("_elapsed_ms", 0)
        for name, res in results.items()
        if "_elapsed_ms" in res
    }

    return {
        "ticker":                  ticker,
        "sector":                  sector,
        "fundamental_score":       fund_score,
        "insider_score":           insider_score,
        "retail_sentiment_score":  retail_score,
        "macro_score":             macro_score,
        "divergence_score":        div_score,
        "composite_score":         composite_score,
        "recommended_action":      _recommended_action(fund_score, retail_score, insider_score, macro_score),
        "behavioral_signal":       behavioral_signal,
        "dominant_retail_narrative": dominant_narrative,
        "top_claims":              [],   # populated by agent if needed
        "key_risks":               key_risks,
        "data_sources_used":       sources_used,
        "failed_sources":          failed_sources,
        "data_freshness_ms":       data_freshness,
        "raw": {
            "edgar_financials": results.get("edgar_financials", {}),
            "edgar_form4":      results.get("edgar_form4", {}),
            "edgar_filings":    results.get("edgar_filings", {}),
            "fred_macro":       results.get("fred_macro", {}),
            "open_insider":     results.get("open_insider", {}),
            "reddit":           results.get("reddit", {}),
            "stocktwits":       results.get("stocktwits", {}),
            "trends":           results.get("trends", {}),
        },
        "_elapsed_ms": round((time.time() - t0) * 1000),
    }
