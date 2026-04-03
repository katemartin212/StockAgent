#!/usr/bin/env python3
"""
server.py — FastAPI backend for the Multi-Sector Research Dashboard

Sector-aware tool routing: detects sector via get_sector_profile, then runs
universal tools + the appropriate sector-specific tool set. Streams SSE events.

Usage:
    uvicorn server:app --reload
Then open dashboard.html in your browser.
"""

import re
import json
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from master_signal import get_master_analysis
from data_sources.comps_data import fetch_comps
from predictive_analytics import run_all_predictive
from validation import run_full_validation

from tools_universal import (
    client,
    get_sector_profile,
    get_stock_price,
    get_company_info,
    get_financial_data,
    get_macro_sensitivity,
    get_insider_activity,
    get_sector_behavioral_biases,
    get_dcf_implied_growth,
    get_dilution_rate,
)
from tools_tech import (
    get_news_sentiment,
    get_earnings_surprise,
    get_reddit_sentiment,
    get_net_revenue_retention,
)
from tools_healthcare import (
    get_pipeline_value,
    get_patent_cliff,
    get_fda_catalyst_risk,
)
from tools_financials import (
    get_net_interest_margin,
    get_loan_loss_provisions,
    get_efficiency_ratio,
)
from tools_consumer import (
    get_same_store_sales,
    get_inventory_turns,
    get_gross_margin_by_channel,
)
from tools_energy import (
    get_break_even_price,
    get_reserve_replacement,
)
from tools_realestate import (
    get_ffo,
    get_cap_rate,
)
from tools_industrials import (
    get_book_to_bill,
    get_capacity_utilization,
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Multi-Sector Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    ticker: str


class CompsRequest(BaseModel):
    subject_ticker: str
    peers: List[str]
    sector: Optional[str] = None


class ValidateRequest(BaseModel):
    ticker: str
    sector: Optional[str] = None


class PrefetchRequest(BaseModel):
    ticker: str
    sources: Optional[List[str]] = None


# ── Tool pipelines ────────────────────────────────────────────────────────────
# Universal ticker-based tools (run for every sector, after sector_profile + biases)

UNIVERSAL_TICKER_PIPELINE = [
    ("get_stock_price",        get_stock_price),
    ("get_company_info",       get_company_info),
    ("get_financial_data",     get_financial_data),
    ("get_macro_sensitivity",  get_macro_sensitivity),
    ("get_insider_activity",   get_insider_activity),
    ("get_dcf_implied_growth", get_dcf_implied_growth),
    ("get_dilution_rate",      get_dilution_rate),
]

# Sector-specific additions (appended after universal)
SECTOR_PIPELINE: dict[str, list] = {
    "Technology": [
        ("get_news_sentiment",        get_news_sentiment),
        ("get_earnings_surprise",     get_earnings_surprise),
        ("get_reddit_sentiment",      get_reddit_sentiment),
        ("get_net_revenue_retention", get_net_revenue_retention),
    ],
    "Communication Services": [
        ("get_news_sentiment",        get_news_sentiment),
        ("get_earnings_surprise",     get_earnings_surprise),
        ("get_reddit_sentiment",      get_reddit_sentiment),
        ("get_net_revenue_retention", get_net_revenue_retention),
    ],
    "Healthcare": [
        ("get_pipeline_value",    get_pipeline_value),
        ("get_patent_cliff",      get_patent_cliff),
        ("get_fda_catalyst_risk", get_fda_catalyst_risk),
        ("get_news_sentiment",    get_news_sentiment),
        ("get_earnings_surprise", get_earnings_surprise),
    ],
    "Financial Services": [
        ("get_net_interest_margin",  get_net_interest_margin),
        ("get_loan_loss_provisions", get_loan_loss_provisions),
        ("get_efficiency_ratio",     get_efficiency_ratio),
        ("get_news_sentiment",       get_news_sentiment),
        ("get_earnings_surprise",    get_earnings_surprise),
    ],
    "Consumer Cyclical": [
        ("get_same_store_sales",        get_same_store_sales),
        ("get_inventory_turns",         get_inventory_turns),
        ("get_gross_margin_by_channel", get_gross_margin_by_channel),
        ("get_news_sentiment",          get_news_sentiment),
        ("get_earnings_surprise",       get_earnings_surprise),
    ],
    "Consumer Defensive": [
        ("get_same_store_sales",        get_same_store_sales),
        ("get_inventory_turns",         get_inventory_turns),
        ("get_gross_margin_by_channel", get_gross_margin_by_channel),
        ("get_news_sentiment",          get_news_sentiment),
        ("get_earnings_surprise",       get_earnings_surprise),
    ],
    "Energy": [
        ("get_break_even_price",    get_break_even_price),
        ("get_reserve_replacement", get_reserve_replacement),
        ("get_news_sentiment",      get_news_sentiment),
        ("get_earnings_surprise",   get_earnings_surprise),
    ],
    "Real Estate": [
        ("get_ffo",               get_ffo),
        ("get_cap_rate",          get_cap_rate),
        ("get_news_sentiment",    get_news_sentiment),
        ("get_earnings_surprise", get_earnings_surprise),
    ],
    "Industrials": [
        ("get_book_to_bill",         get_book_to_bill),
        ("get_capacity_utilization", get_capacity_utilization),
        ("get_news_sentiment",       get_news_sentiment),
        ("get_earnings_surprise",    get_earnings_surprise),
    ],
    "_default": [
        ("get_news_sentiment",    get_news_sentiment),
        ("get_earnings_surprise", get_earnings_surprise),
    ],
}

# Human-readable toolkit names for dashboard display
TOOLKIT_NAMES = {
    "Technology":             "Technology Toolkit",
    "Communication Services": "Communications Toolkit",
    "Healthcare":             "Healthcare Toolkit",
    "Financial Services":     "Financial Services Toolkit",
    "Consumer Cyclical":      "Consumer Cyclical Toolkit",
    "Consumer Defensive":     "Consumer Defensive Toolkit",
    "Energy":                 "Energy Toolkit",
    "Real Estate":            "Real Estate / REIT Toolkit",
    "Industrials":            "Industrials Toolkit",
    "_default":               "Universal Toolkit",
}


# ── Synthesis prompt ──────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are a senior equity analyst at a multi-sector investment fund.
You will receive LIVE data from multiple research tools for a stock ticker.
Synthesize it into ONLY a valid JSON object — no markdown fences, no text outside the JSON.

Required JSON structure:
{
  "ticker": string,
  "company_name": string,
  "sector": string,
  "industry": string,
  "sector_toolkit": string (e.g. "Healthcare Toolkit (13 tools)"),
  "price": number,
  "price_change_pct": number,
  "market_cap_b": number,
  "price_52w_high": number or null,
  "price_52w_low": number or null,
  "avg_volume_m": number or null (average daily volume in millions, from get_company_info or get_stock_price),
  "analyst_target": number or null (copy from fundamentals for convenience),
  "company_overview": {
    "description": "2-3 sentence plain-English description of what the company does and how it makes money — from get_company_info longBusinessSummary, summarised and clarified",
    "founded": "founding year as string, or null if unknown",
    "headquarters": "city, state/country",
    "employees": "approximate headcount as formatted string e.g. '92,000'",
    "business_model": "one tight sentence: how the company generates revenue (subscription, transactional, licensing, etc.)",
    "key_products": ["up to 4 flagship products or services by name"],
    "competitive_position": "one sentence on the company's moat, market share, or key competitive advantage",
    "competitors": ["up to 5 main competitor ticker symbols — must be real publicly-traded US equities, e.g. AMD"],
    "next_earnings_date": "YYYY-MM-DD string or null",
    "ex_dividend_date": "YYYY-MM-DD string or null",
    "revenue_ttm_b": "number or null (trailing 12-month revenue in billions)",
    "employees_change_pct": "number or null (YoY headcount change %, e.g. -3.2)"
  },
  "scores": {
    "fundamental": integer 0-100,
    "retail_sentiment": integer 0-100,
    "divergence": integer 0-100,
    "composite_signal": integer 0-100
  },
  "macro": {
    "regime": string (e.g. "Easing", "Tightening", "Stable"),
    "fed_funds_pct": number or null,
    "treasury_10y_pct": number or null,
    "spread_2s10s": number or null,
    "yield_curve_inverted": boolean,
    "cpi_yoy_pct": number or null,
    "plain_english": string
  },
  "social": {
    "reddit_sentiment": string ("bullish"|"bearish"|"neutral"),
    "reddit_posts": integer,
    "stocktwits_score": integer (0-100),
    "stocktwits_watchers": integer,
    "search_interest": integer (0-100) or null,
    "near_search_peak": boolean,
    "signal_note": string (copy from master_signal.raw.reddit.signal_note — shows filter stats or fallback state),
    "low_coverage": boolean (true if < 3 relevant posts found),
    "top_reddit_posts": [ { "title": string, "subreddit": string, "score": integer, "sentiment": string, "relevance_score": integer, "flair": string or null, "top_comment": string or null } ],
    "top_stocktwits": [ { "body": string, "sentiment": string, "username": string } ]
  },
  "insider_deep": {
    "edgar_weighted_signal": number or null,
    "cluster_buy": boolean,
    "net_flow_usd": number or null,
    "recent_transactions": [ { "name": string, "type": string, "value_usd": integer, "date": string } ]
  },
  "behavioral_signal": {
    "state": "Peak Euphoria|Distribution|Accumulation|Capitulation|Recovery|Aligned",
    "explanation": "one institutional-grade sentence grounded in sector behavioral biases data"
  },
  "retail_narrative": [
    {
      "claim": "specific claim retail investors commonly make (draw from get_sector_behavioral_biases)",
      "verdict": "Supported|Misleading|Unsupported",
      "explanation": "fact-check using exact numbers from the provided live data"
    }
  ],
  "fundamentals": {
    "revenue_growth_pct": number,
    "gross_margin_pct": number,
    "fcf_margin_pct": number,
    "rule_of_40": number,
    "pe_ratio": number,
    "forward_pe": number,
    "ev_revenue": number,
    "short_interest_pct": number,
    "insider_activity": string (from get_insider_activity: combine filings_last_90_days, insider_signal, top filer names),
    "analyst_consensus": string,
    "analyst_target": number,
    "earnings_beat_rate": string,
    "implied_revenue_cagr_pct": number or null,
    "trailing_revenue_cagr_pct": number or null,
    "dcf_gap_pct": number or null,
    "dcf_plain_english": string or null,
    "sbc_pct_revenue": number or null,
    "sbc_adjusted_fcf_margin_pct": number or null,
    "share_count_growth_pct": number or null,
    "dilution_flag": string or null,
    "nrr_applicable": boolean,
    "nrr_pct": number or null,
    "nrr_trend": string or null,
    "nrr_benchmark": string or null,
    "macro_rate_sensitivity": string (score + explanation from get_macro_sensitivity),
    "macro_usd_sensitivity": string,
    "macro_gdp_sensitivity": string
  },
  "sector_metrics": [
    {
      "label": string,
      "value": string,
      "signal": "positive|negative|neutral|warning|null"
    }
  ],
  "reasoning_trace": [
    {
      "step": integer,
      "tool": "exact tool function name",
      "input": "ticker or sector",
      "output_summary": "key numbers from the actual data provided",
      "reasoning": "what this data revealed about the stock"
    }
  ],
  "verdict": "ACCUMULATE|HOLD|TRIM|AVOID",
  "verdict_rationale": "2 sentence institutional rationale grounded in the live data"
}

FIELD MAPPINGS — use exact data from tools:
- get_company_info: populate company_overview.description from longBusinessSummary (summarise to 2-3 sentences max),
  company_overview.founded from founded/ipoDate, company_overview.headquarters from city+country,
  company_overview.employees from fullTimeEmployees, company_overview.key_products from products/services mentioned,
  company_overview.competitors from known public peers (use sector knowledge — real ticker symbols only),
  company_overview.next_earnings_date from earningsDate (format YYYY-MM-DD), ex_dividend_date from exDividendDate,
  company_overview.revenue_ttm_b from totalRevenue/1e9, company_overview.employees_change_pct from any YoY headcount data
- get_sector_profile: sector, industry, company_name
- get_sector_behavioral_biases: use the biases array to construct retail_narrative claims
- get_macro_sensitivity: map sensitivity.interest_rates → macro_rate_sensitivity (score + explanation),
  sensitivity.usd_strength → macro_usd_sensitivity, sensitivity.gdp_cycle → macro_gdp_sensitivity
- get_insider_activity: map filings_last_90_days + insider_signal + recent_filings[].filer → insider_activity string
  e.g. "3 Form 4 filings (90d) · moderate activity · Smith (CFO), Jones (CEO)"
- get_dcf_implied_growth: map implied_10yr_revenue_cagr_pct → implied_revenue_cagr_pct,
  trailing_3yr_revenue_cagr_pct → trailing_revenue_cagr_pct, gap_pct → dcf_gap_pct,
  plain_english → dcf_plain_english. Set all to null if tool returned an error.
- get_dilution_rate: map sbc_pct_revenue, sbc_adjusted_fcf_margin_pct,
  share_count_growth_pct, dilution_flag directly. Set to null if tool returned an error.
- get_net_revenue_retention: set nrr_applicable from tool's nrr_applicable field.
  If true, map nrr_estimate_pct → nrr_pct, trend → nrr_trend, benchmark → nrr_benchmark.
  If false or error, set nrr_applicable: false, nrr_pct/nrr_trend/nrr_benchmark to null.

SECTOR_METRICS — populate based on sector-specific tool data:
- Technology/Communication Services:
  NRR Estimate (%), Earnings Beat Rate, News Sentiment Score, Reddit Sentiment
- Healthcare:
  R&D Intensity (% revenue), Pipeline Rating, Patent Cliff Risk, FDA Catalyst Risk Score
- Financial Services:
  Net Interest Margin (%), NIM Trend, Efficiency Ratio (%), Loan Loss Provision Trend
- Consumer Cyclical/Defensive:
  Same-Store Sales Proxy (%), Inventory Turns, GM Trend
- Energy:
  Cost/Revenue Coverage (%), Break-Even Risk Level, Reserve Replacement Signal
- Real Estate:
  P/FFO Multiple, FFO Yield (%), Implied Cap Rate (%)
- Industrials:
  Book-to-Bill (implied), Capacity Utilization Signal, Operating Leverage Ratio
For signal field: "positive" = good/green, "negative" = bad/red, "warning" = caution/amber, "neutral" = neutral/gray

MASTER_SIGNAL FIELD MAPPINGS — from master_signal data:
- master_signal.composite_score → scores.composite_signal
- master_signal.macro_score and raw.fred_macro → populate macro{} block
- master_signal.raw.reddit → social.reddit_sentiment, reddit_posts, top_reddit_posts, signal_note, low_coverage
  Use reddit.posts_passed_filter (not posts_found) as the authoritative post count — it excludes irrelevant posts.
  If reddit.low_coverage is true, set social.reddit_posts = 0 and use the fallback_state in behavioral_signal reasoning.
- master_signal.raw.stocktwits → social.stocktwits_score, stocktwits_watchers, top_stocktwits
- master_signal.raw.trends → social.search_interest, near_search_peak
- master_signal.raw.edgar_form4 → populate insider_deep{} block
- master_signal.behavioral_signal → use to inform behavioral_signal.state
- master_signal.dominant_retail_narrative → use as additional evidence in retail_narrative claims
- master_signal.key_risks → mention in verdict_rationale if relevant

SCORING:
- fundamental (0-100): Start at 50.
  Revenue growth > 20%: +15; > 10%: +8; < 0%: -15; < -5%: -25
  Gross margin > 60%: +15; > 40%: +8; < 20%: -10
  FCF margin > 20%: +15; > 10%: +8; < 0%: -20
  For REITs: add +10 if FFO yield > 5%, +10 if P/FFO < 16×
  For banks: add +10 if NIM > 3%, +10 if efficiency ratio < 58%
  For Industrials: add +10 if book-to-bill > 1.0
  Cap at 100, floor at 0.
- retail_sentiment (0-100): 50 = neutral baseline.
  Blend news sentiment (from get_news_sentiment) with master_signal.retail_sentiment_score if available.
  Weight: news 50%, master_signal retail 50%.
- divergence (0-100): min(abs(fundamental - retail_sentiment) * 1.5, 100).
- composite_signal: use master_signal.composite_score if available, else compute:
  (fundamental * 0.40 + master_signal.insider_score * 0.25 + retail_sentiment * 0.20 + macro_score * 0.15).

BEHAVIORAL SIGNAL:
- Peak Euphoria:  retail_sentiment > 70 AND fundamental < 50
- Distribution:   retail_sentiment > 60 AND fundamental between 40-65
- Accumulation:   retail_sentiment < 40 AND fundamental > 55
- Capitulation:   retail_sentiment < 30 AND fundamental > 60
- Aligned:        divergence < 20
- Recovery:       everything else

Include 4-5 retail_narrative claims grounded in the sector behavioral biases data.
Include one reasoning_trace step per tool used. Return ONLY the raw JSON object."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def sse(payload: dict) -> str:
    """Encode a dict as a single SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _sanitize_json(text: str) -> str:
    """
    Escape bare control characters (newlines, tabs, carriage returns) that
    appear inside JSON string values. LLMs occasionally emit these literally
    instead of as \\n / \\t, which causes json.loads to fail with a
    'Expecting ,' delimiter error even when the structure is otherwise valid.
    """
    result = []
    in_string = False
    escape_next = False
    _escapes = {'\n': '\\n', '\r': '\\r', '\t': '\\t'}
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == '\\' and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch in _escapes:
            result.append(_escapes[ch])
        else:
            result.append(ch)
    return ''.join(result)


def extract_json(text: str) -> str:
    """Strip markdown fences, sanitize control chars, return the JSON object."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fenced:
        return _sanitize_json(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return _sanitize_json(text[start : end + 1])
    return text


# ── Core streaming generator ──────────────────────────────────────────────────

def _stream_analysis(ticker: str):
    """
    Synchronous generator: runs sector detection, then universal + sector-specific
    tools, then Claude synthesis. Yields SSE events throughout.

    SSE event types:
      tool_start      → tool is about to run
      tool_done       → tool finished
      sector_detected → sector identified (fires once after get_sector_profile)
      synthesizing    → Claude synthesis starting
      result          → final JSON analysis
      error           → something went wrong
    """
    import time as _time
    _t_stream_start = _time.time()
    _perf: list[dict] = []   # performance_breakdown accumulator

    gathered = {}
    sector = None

    # ── Step 1: Sector profile (always first) ─────────────────────────────────
    yield sse({"type": "tool_start", "tool": "get_sector_profile"})
    _t0 = _time.time()
    try:
        raw = get_sector_profile(ticker)
        data = json.loads(raw)
        gathered["get_sector_profile"] = data
        sector = data.get("sector")
        _perf.append({"source": "get_sector_profile", "duration_ms": round((_time.time() - _t0) * 1000), "cache_hit": data.get("_cached", False)})
        yield sse({"type": "tool_done", "tool": "get_sector_profile"})
    except Exception as e:
        _perf.append({"source": "get_sector_profile", "duration_ms": round((_time.time() - _t0) * 1000), "cache_hit": False, "error": str(e)})
        gathered["get_sector_profile"] = {"error": str(e)}
        yield sse({"type": "tool_done", "tool": "get_sector_profile", "error": str(e)})

    # Determine toolkit and broadcast sector info
    sector_key   = sector if sector in SECTOR_PIPELINE else "_default"
    toolkit_name = TOOLKIT_NAMES.get(sector_key, "Universal Toolkit")
    total_tools  = 1 + 1 + len(UNIVERSAL_TICKER_PIPELINE) + len(SECTOR_PIPELINE[sector_key]) + 1  # +1 synthesis
    toolkit_label = f"{toolkit_name} ({total_tools} tools)"

    yield sse({
        "type":    "sector_detected",
        "sector":  sector or "Unknown",
        "industry": gathered.get("get_sector_profile", {}).get("industry", ""),
        "toolkit": toolkit_label,
    })

    # ── Step 2: Sector behavioral biases (needs sector, not ticker) ───────────
    yield sse({"type": "tool_start", "tool": "get_sector_behavioral_biases"})
    _t0 = _time.time()
    try:
        raw = get_sector_behavioral_biases(sector or "_default")
        gathered["get_sector_behavioral_biases"] = json.loads(raw)
        _perf.append({"source": "get_sector_behavioral_biases", "duration_ms": round((_time.time() - _t0) * 1000), "cache_hit": False})
        yield sse({"type": "tool_done", "tool": "get_sector_behavioral_biases"})
    except Exception as e:
        _perf.append({"source": "get_sector_behavioral_biases", "duration_ms": round((_time.time() - _t0) * 1000), "cache_hit": False, "error": str(e)})
        gathered["get_sector_behavioral_biases"] = {"error": str(e)}
        yield sse({"type": "tool_done", "tool": "get_sector_behavioral_biases", "error": str(e)})

    # ── Steps 3+4: Universal + sector-specific tools — all parallel ──────────
    # None of these tools depend on each other's output, so all can fire at once.
    # We emit tool_start for all immediately, then tool_done as each completes.
    all_tools = list(UNIVERSAL_TICKER_PIPELINE) + list(SECTOR_PIPELINE[sector_key])

    for tool_name, _ in all_tools:
        yield sse({"type": "tool_start", "tool": tool_name})

    import concurrent.futures as _cf2
    _t0_tools = _time.time()

    def _run_tool(tool_name, tool_fn):
        t0 = _time.time()
        try:
            raw = tool_fn(ticker)
            result = json.loads(raw)
            ms = round((_time.time() - t0) * 1000)
            return tool_name, result, ms, None
        except Exception as e:
            ms = round((_time.time() - t0) * 1000)
            return tool_name, {"error": str(e)}, ms, str(e)

    with _cf2.ThreadPoolExecutor(max_workers=len(all_tools)) as _tool_pool:
        _tool_futures = {
            _tool_pool.submit(_run_tool, name, fn): name
            for name, fn in all_tools
        }
        for _fut in _cf2.as_completed(_tool_futures, timeout=60):
            tool_name, result, ms, err = _fut.result()
            gathered[tool_name] = result
            _perf.append({"source": tool_name, "duration_ms": ms, "cache_hit": result.get("_cached", False), **({"error": err} if err else {})})
            yield sse({"type": "tool_done", "tool": tool_name, **({"error": err} if err else {})})

    # ── Step 5: Master analysis → behavioral inputs → predictive analytics ──────
    # master_analysis runs first (almost always cached, sub-second) so that
    # divergence_score, macro_score, and insider_signal are available as
    # behavioral inputs to the DCF scenario model in run_all_predictive.
    master_data = None
    predictive_data = None
    yield sse({"type": "parallel_start", "sources": [
        "SEC EDGAR", "FRED Macro", "Reddit", "StockTwits", "Google Trends"
    ]})
    _t0_parallel = _time.time()
    try:
        master_data = get_master_analysis(ticker, sector)

        # Build behavioral_inputs from master_data signals for the scenario model.
        # divergence_score from master_signal: HIGH = strong fundamentals / low retail
        # attention (undervalued). We flip it so HIGH = overhyped, matching the
        # spec convention where high divergence → narrative premium discount.
        _insider_score = master_data.get("insider_score", 50)
        _behavioral_inputs = {
            "divergence_score": 100.0 - float(master_data.get("divergence_score", 50)),
            "macro_score":       float(master_data.get("macro_score", 50)),
            "insider_signal": (
                "strongly_bullish" if _insider_score >= 75 else
                "bullish"          if _insider_score >= 60 else
                "strongly_bearish" if _insider_score <= 25 else
                "bearish"          if _insider_score <= 40 else
                "neutral"
            ),
        }
        predictive_data = run_all_predictive(ticker, sector,
                                             behavioral_inputs=_behavioral_inputs)
        # Record individual source timings from master_signal's data_freshness_ms
        for src_name, src_ms in (master_data.get("data_freshness_ms") or {}).items():
            _perf.append({"source": f"master:{src_name}", "duration_ms": src_ms, "cache_hit": src_ms == 0})
        _perf.append({"source": "run_all_predictive", "duration_ms": round((_time.time() - _t0_parallel) * 1000), "cache_hit": False})
        # Emit source_done for each source that returned data
        source_map = {
            "edgar_financials": "SEC EDGAR (Financials)",
            "edgar_form4":      "SEC EDGAR (Form 4)",
            "edgar_filings":    "SEC EDGAR (Filings)",
            "fred_macro":       "FRED Macro",
            "reddit":           "Reddit",
            "stocktwits":       "StockTwits",
            "trends":           "Google Trends",
        }
        for key, label in source_map.items():
            raw_src = (master_data.get("raw") or {}).get(key) or {}
            has_error = "error" in raw_src
            yield sse({
                "type":   "source_done",
                "source": label,
                "ok":     not has_error,
                "note":   raw_src.get("signal_note") or raw_src.get("error") or "",
            })
        yield sse({
            "type":             "parallel_done",
            "sources_ok":       len(master_data.get("data_sources_used", [])),
            "sources_failed":   len(master_data.get("failed_sources", [])),
            "composite_score":  master_data.get("composite_score"),
            "behavioral_signal": master_data.get("behavioral_signal"),
        })
        # Emit predictive analytics data for the Predict tab (not passed to Claude)
        if predictive_data:
            yield sse({"type": "predictive_done", "data": predictive_data})
        # Pass a trimmed summary (not the full raw data) to keep the prompt manageable
        raw = master_data.get("raw", {})
        gathered["master_signal"] = {k: v for k, v in master_data.items() if k != "raw"}
        gathered["master_signal_enrichment"] = {
            "macro": {
                k: v for k, v in raw.get("fred_macro", {}).items()
                if k not in ("errors", "data_source", "_elapsed_ms", "as_of")
            },
            "reddit": {
                k: raw.get("reddit", {}).get(k)
                for k in ("overall_sentiment", "sentiment_score", "weighted_score",
                           "posts_found", "total_fetched", "posts_passed_filter",
                           "low_coverage", "fallback_state",
                           "subreddit_breakdown", "sentiment_breakdown",
                           "engagement_score", "signal_note", "auth_method")
            },
            "reddit_top3": raw.get("reddit", {}).get("top_posts", [])[:3],
            "stocktwits": {
                k: raw.get("stocktwits", {}).get(k)
                for k in ("overall_sentiment", "sentiment_score", "bullish_count",
                           "bearish_count", "watchers", "messages_analyzed", "signal_note")
            },
            "stocktwits_top3": raw.get("stocktwits", {}).get("top_messages", [])[:3],
            "trends": {
                k: raw.get("trends", {}).get(k)
                for k in ("current_interest", "avg_3mo", "peak_value", "peak_date",
                           "near_peak", "trend_vs_avg", "signal_note", "related_queries")
            },
        }
    except Exception as e:
        yield sse({"type": "parallel_done", "error": str(e)})

    # ── Step 6: Claude synthesizes all gathered data ──────────────────────────
    _t0_synthesis = _time.time()
    yield sse({"type": "synthesizing"})

    user_msg = (
        f"Ticker: {ticker}\n"
        f"Detected sector: {sector or 'Unknown'}\n"
        f"Toolkit applied: {toolkit_label}\n\n"
        f"Live data from research tools:\n"
        f"{json.dumps(gathered, indent=2)}"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_text = response.content[0].text
        json_str = extract_json(raw_text)

        try:
            analysis = json.loads(json_str)
        except json.JSONDecodeError as first_err:
            # Log the exact failure point so we can diagnose the root cause
            import logging as _logging
            _logging.getLogger("stock_agent").error(
                f"JSON parse failed for {ticker} at {first_err}. "
                f"Offending area: ...{json_str[max(0, first_err.pos-80):first_err.pos+80]}..."
            )
            # Self-repair: send Claude its own malformed output and ask for a fix
            repair_response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                messages=[{
                    "role": "user",
                    "content": (
                        f"The following JSON is malformed. Error: {first_err}\n\n"
                        f"Return ONLY the corrected valid JSON object — no explanation, no fences:\n\n"
                        f"{json_str[:12000]}"
                    ),
                }],
            )
            repaired = extract_json(repair_response.content[0].text)
            analysis = json.loads(repaired)

        if not analysis.get("sector_toolkit"):
            analysis["sector_toolkit"] = toolkit_label
        _perf.append({"source": "claude_synthesis", "duration_ms": round((_time.time() - _t0_synthesis) * 1000), "cache_hit": False})
        analysis["performance_breakdown"] = _perf
        analysis["total_analysis_ms"] = round((_time.time() - _t_stream_start) * 1000)
        yield sse({"type": "result", "data": analysis})

    except json.JSONDecodeError as e:
        yield sse({"type": "error", "message": f"JSON parse error (repair also failed): {e}. Try again."})
    except Exception as e:
        yield sse({"type": "error", "message": str(e)})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/cache/stats")
def get_cache_stats():
    from data_sources._cache import cache_stats
    stats = cache_stats()
    stats["display"] = f"Cache saved ~{stats['saved_s']}s this session ({stats['hit_rate_pct']}% hit rate)"
    return JSONResponse(content=stats)


def _prefetch_background(ticker: str, sources: list[str]):
    """Fire slow data fetches in background threads. Returns immediately."""
    import importlib
    _source_map = {
        "edgar":   [("data_sources.sec_edgar", "get_edgar_financials"),
                    ("data_sources.sec_edgar", "get_edgar_form4"),
                    ("data_sources.sec_edgar", "get_edgar_filings")],
        "fred":    [("data_sources.fred_macro", "get_fred_macro")],
        "profile": [("tools_universal", "get_sector_profile"),
                    ("tools_universal", "get_company_info")],
        "reddit":  [("data_sources.reddit_sentiment", "get_reddit_sentiment")],
        "trends":  [("data_sources.trends_signal", "get_search_interest")],
    }
    tasks = []
    for src in (sources or ["edgar", "fred", "profile"]):
        tasks.extend(_source_map.get(src, []))

    def _call(mod_path, fn_name):
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, fn_name)
            fn(ticker) if fn_name != "get_fred_macro" else fn()
        except Exception:
            pass

    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for mod_path, fn_name in tasks:
            pool.submit(_call, mod_path, fn_name)


@app.post("/prefetch")
def prefetch(req: PrefetchRequest):
    """Fire slow fetches in background. Returns 202 immediately — never blocks."""
    ticker = req.ticker.upper().strip()
    import threading
    t = threading.Thread(
        target=_prefetch_background,
        args=(ticker, req.sources),
        daemon=True,
    )
    t.start()
    return JSONResponse(status_code=202, content={"status": "prefetch_started", "ticker": ticker})


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    ticker = req.ticker.upper().strip()
    return StreamingResponse(
        _stream_analysis(ticker),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


def _generate_comps_verdict(comps: dict) -> str:
    """Call Claude to write a 2-sentence comps verdict from the table data."""
    subject = comps.get("subject_ticker", "")
    rows = comps.get("rows", [])
    median = comps.get("median", {})

    # Build a compact table summary for the prompt
    cols = ["ev_revenue", "ev_ebitda", "forward_pe", "revenue_growth_pct", "gross_margin_pct"]
    lines = []
    for r in rows:
        vals = ", ".join(
            f"{c}={r[c]}" for c in cols if r.get(c) is not None
        )
        tag = " [SUBJECT]" if r.get("is_subject") else (" [ETF]" if r.get("is_etf") else "")
        lines.append(f"  {r['ticker']}{tag}: {vals}")
    if median:
        vals = ", ".join(f"{c}={median[c]}" for c in cols if median.get(c) is not None)
        lines.append(f"  MEDIAN: {vals}")

    # vs-median premium/discount for subject
    subject_row = next((r for r in rows if r.get("is_subject")), {})
    vs_lines = []
    for col in ("ev_revenue", "ev_ebitda", "forward_pe"):
        key = f"{col}_vs_median_pct"
        if subject_row.get(key) is not None:
            vs_lines.append(f"{col}: {subject_row[key]:+.1f}% vs median")

    prompt = (
        f"Peer comparison table for {subject}:\n"
        + "\n".join(lines)
        + (f"\n\nSubject vs median: {', '.join(vs_lines)}" if vs_lines else "")
        + "\n\nWrite exactly 2 sentences: first, state whether "
          f"{subject} trades at a premium or discount to peers and by how much "
          "(cite specific multiples); second, assess whether the premium/discount "
          "is justified given relative growth and margins. Be concise and institutional."
    )

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logging.getLogger("stock_agent").warning(f"comps verdict error: {e}")
        return ""


@app.post("/comps")
def comps(req: CompsRequest):
    try:
        data = fetch_comps(req.subject_ticker, req.peers, req.sector)
        verdict = _generate_comps_verdict(data)
        data["verdict"] = verdict
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/validate")
def validate(req: ValidateRequest):
    """
    Run walk-forward validation for all three predictive models.
    Returns tier (HIGH/MEDIUM/LOW/UNVALIDATED) and per-model cards.
    This endpoint is compute-heavy — expect 60-180s for cold requests.
    """
    ticker = req.ticker.upper().strip()
    try:
        result = run_full_validation(ticker, req.sector)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
