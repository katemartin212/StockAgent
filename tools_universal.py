#!/usr/bin/env python3
"""
tools_universal.py
------------------
Tools that run for every ticker regardless of sector:
  - calculator
  - get_stock_price
  - get_company_info
  - get_financial_data
  - get_dcf_implied_growth
  - get_dilution_rate
  - get_sector_profile     (NEW — always the first tool; drives routing)
  - get_macro_sensitivity  (NEW — rates/USD/GDP sensitivity scores)
  - get_insider_activity   (NEW — SEC EDGAR Form 4 API, no scraping)
  - get_sector_behavioral_biases (NEW — sector-specific retail misconceptions)
"""

import os
import re
import json
import time
from datetime import datetime, timedelta

import requests
import yfinance as yf
import anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Shared client ──────────────────────────────────────────────────────────────

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set.\n"
        "Add it to .env: ANTHROPIC_API_KEY=sk-ant-...\n"
        "Or export it: export ANTHROPIC_API_KEY=sk-ant-..."
    )

client = anthropic.Anthropic(api_key=api_key)
_today = datetime.now().strftime("%B %d, %Y")

# ── yfinance field allowlist ────────────────────────────────────────────────────
# yfinance .info returns 100+ fields; we filter to only what the pipeline uses.
# This avoids sending massive payloads through the tool chain.

_YF_FIELDS = {
    "longName", "shortName", "sector", "industry",
    "currentPrice", "regularMarketPrice", "previousClose",
    "regularMarketChangePercent", "regularMarketChange",
    "marketCap", "enterpriseValue",
    "trailingPE", "forwardPE", "priceToBook",
    "enterpriseToRevenue", "enterpriseToEbitda",
    "grossMargins", "operatingMargins", "profitMargins",
    "revenueGrowth", "earningsGrowth",
    "totalRevenue", "grossProfits", "freeCashflow",
    "operatingCashflow", "ebitda",
    "shortPercentOfFloat", "shortRatio",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice",
    "recommendationKey", "numberOfAnalystOpinions",
    "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "averageVolume", "averageVolume10days", "volume",
    "sharesOutstanding", "floatShares",
    "dividendYield", "exDividendDate",
    "longBusinessSummary", "fullTimeEmployees",
    "city", "state", "country",
    "website", "phone",
    "earningsDate", "nextFiscalYearEnd", "mostRecentQuarter",
    "founded", "ipoDate",
    "totalDebt", "totalCash", "debtToEquity",
    "returnOnEquity", "returnOnAssets",
    "revenuePerShare", "bookValue",
    "heldPercentInstitutions", "heldPercentInsiders",
}


def _filter_info(raw_info: dict) -> dict:
    """Return only the fields in the allowlist to reduce payload size."""
    return {k: v for k, v in raw_info.items() if k in _YF_FIELDS}

# ── Sector behavioral bias lookup ──────────────────────────────────────────────
# Top 3 retail misconceptions per sector. Used by get_sector_behavioral_biases.

_SECTOR_BIASES = {
    "Technology": [
        "Retail investors focus on revenue growth while ignoring stock-based compensation "
        "that can exceed 15–20% of revenue, making reported FCF significantly overstated.",
        "High P/E is dismissed as 'justified by growth' without analyzing the DCF-implied CAGR — "
        "which often requires exceptional, historically rare growth rates to hold.",
        "NRR below 110% signals existing customers aren't expanding spend, but retail anchors "
        "to total ARR growth and misses this early deterioration signal.",
    ],
    "Communication Services": [
        "Subscriber count is treated as a proxy for value without analyzing ARPU trends, "
        "churn cohorts, or the increasing cost of content to retain subscribers.",
        "Digital advertising is assumed to be high-margin without accounting for TAC "
        "(traffic acquisition costs) which can consume 20–30% of gross revenue.",
        "Platform network effects are over-valued in retail narratives — "
        "switching costs are often lower than assumed once a competing platform achieves critical mass.",
    ],
    "Healthcare": [
        "Retail investors price drugs as if Phase 3 FDA approval is near-certain — "
        "in reality 50–60% of Phase 3 trials fail, creating binary event risk.",
        "Pipeline assets are often double-counted: a platform company's multiple already includes "
        "some pipeline optionality, which retail then adds again when counting individual programs.",
        "Patent cliff revenue at risk is chronically underestimated — "
        "a single blockbuster losing exclusivity can eliminate 30–50% of earnings overnight.",
    ],
    "Financial Services": [
        "Headline EPS is taken at face value without recognizing that banks can release "
        "loan loss reserves to boost earnings, masking underlying credit deterioration.",
        "Net Interest Margin is only analyzed in the direction that benefits the narrative — "
        "retail misses that rising rates also reprice deposits, eventually compressing NIM.",
        "Credit cycle risk is systematically underestimated in benign periods — "
        "net charge-offs are a lagging indicator and stress appears first in provision trends.",
    ],
    "Consumer Cyclical": [
        "Total revenue growth is used as a proxy for organic demand without stripping out "
        "new store/location openings — same-store sales can be declining while topline grows.",
        "Inventory build signals are ignored until they force markdowns and margin compression "
        "two to three quarters later, by which time the market has already reacted.",
        "Discretionary consumer spending is extrapolated forward in cycles — "
        "retail anchors to post-reopening or stimulus-driven spending as the new baseline.",
    ],
    "Consumer Defensive": [
        "Dividend yield is assumed sustainable without stress-testing payout ratios "
        "against earnings and FCF cyclicality.",
        "Volume/price mix is ignored — revenue growth driven purely by pricing signals "
        "demand elasticity risk and is structurally weaker than volume-driven growth.",
        "Private label competition eroding brand premium is chronically underweighted — "
        "category leaders can see 200–400bps of gross margin pressure over a cycle.",
    ],
    "Energy": [
        "Retail anchors to current or recent-peak commodity prices and projects them "
        "indefinitely, ignoring supply response and mean reversion in oil/gas cycles.",
        "Reserve replacement ratios are rarely analyzed — a company can report strong "
        "current earnings while systematically depleting the asset base that generates them.",
        "Break-even prices and capex cycles are ignored — retail focuses on current "
        "cash flow without stress-testing profitability at mid-cycle commodity prices.",
    ],
    "Real Estate": [
        "P/E is used to value REITs — this is incorrect; depreciation makes GAAP earnings "
        "meaningless for real estate; P/FFO is the correct multiple.",
        "Occupancy rates and lease duration are rarely analyzed — a fully-leased REIT "
        "with short-term leases has very different roll risk than long-term contracted cash flows.",
        "Cap rate compression during low-rate environments is mistaken for fundamental "
        "value improvement — it is often a liquidity/discount rate artifact that reverses.",
    ],
    "Industrials": [
        "Book-to-bill ratios and backlog are ignored in favor of current-quarter revenue, "
        "causing retail to miss both acceleration and deceleration signals 2–3 quarters early.",
        "Capacity utilization near peak creates capex catch-up requirements that "
        "compress future FCF — retail mistakes high utilization as pure margin positive.",
        "Input cost pressures (materials, labor, freight) hit margins with a lag relative "
        "to when they appear in commodity indices, causing earnings surprises.",
    ],
    "Basic Materials": [
        "Commodity price moves are extrapolated into future earnings without modeling "
        "producer supply response and price mean reversion.",
        "Operating leverage in mining/materials is underappreciated — "
        "a 10% commodity price decline can eliminate 40–60% of EBIT at high-cost producers.",
        "Capital intensity and long project lead times are ignored — "
        "capex decisions made at cycle peaks destroy value when prices fall before completion.",
    ],
    "Utilities": [
        "Utilities are treated as low-risk without analyzing rate case outcomes — "
        "regulatory commissions can cap returns below cost of capital.",
        "Dividend yield chasing ignores balance sheet leverage — "
        "many utilities carry 4–6× net debt/EBITDA and are acutely sensitive to rate rises.",
        "Capex for grid modernization is misread as a negative (cash burn) "
        "rather than a positive (growing regulated asset base earning allowed returns).",
    ],
    "_default": [
        "Retail investors frequently extrapolate recent earnings momentum without "
        "stress-testing the business model at mid-cycle conditions.",
        "Valuation multiples are justified by narrative rather than anchored to "
        "a specific growth assumption — the DCF-implied CAGR is rarely calculated.",
        "Management guidance is taken at face value without tracking beat/miss history "
        "to calibrate how conservative or optimistic the company tends to be.",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS & FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Calculator ─────────────────────────────────────────────────────────────────

calculator_tool = {
    "name": "calculator",
    "description": (
        "Precise arithmetic for financial math: +, -, *, /, ** and parentheses. "
        "Use this for every numerical calculation — never do mental math."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "e.g. '242.58 * 15120000000' or '(385 / 164000) * 1000000'"
            }
        },
        "required": ["expression"]
    }
}

def calculator(expression: str) -> str:
    try:
        if not all(c in "0123456789+-*/.() " for c in expression):
            return "Error: only numbers and +-*/.() are allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# ── Sector Profile — always the first tool; drives routing ────────────────────

sector_profile_tool = {
    "name": "get_sector_profile",
    "description": (
        "ALWAYS call this first for any analysis. Returns the company's sector, "
        "industry, market cap, and a brief business description from yfinance. "
        "The sector field determines which analytical toolkit will be applied."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'JNJ'"}
        },
        "required": ["ticker"]
    }
}

def get_sector_profile(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        info = _filter_info(yf.Ticker(ticker).info)
        desc = info.get("longBusinessSummary") or ""
        return json.dumps({
            "ticker": ticker,
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "description_snippet": desc[:350] + ("…" if len(desc) > 350 else ""),
            "data_source": "Yahoo Finance via yfinance",
        })
    except Exception as e:
        return json.dumps({"error": f"Sector profile failed: {e}"})


# ── Stock Price ────────────────────────────────────────────────────────────────

stock_price_tool = {
    "name": "get_stock_price",
    "description": (
        "Live stock price, daily change %, 52-week range, volume, and market cap. "
        "Use get_financial_data for valuation metrics."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'"}
        },
        "required": ["ticker"]
    }
}

def get_stock_price(ticker: str) -> str:
    try:
        info = _filter_info(yf.Ticker(ticker.upper().strip()).info)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return json.dumps({"error": f"No price data for '{ticker}'."})
        return json.dumps({
            "ticker": ticker.upper(),
            "name": info.get("longName"),
            "price": price,
            "change_pct_today": round(info.get("regularMarketChangePercent") or 0, 2),
            "high_52w": info.get("fiftyTwoWeekHigh"),
            "low_52w": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume"),
            "avg_volume_30d": info.get("averageVolume"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "market_cap": info.get("marketCap"),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Company Info ───────────────────────────────────────────────────────────────

company_info_tool = {
    "name": "get_company_info",
    "description": (
        "Company profile: sector, industry, employee count, CEO, headquarters, "
        "and a brief business description. Use ticker symbol as input."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'MSFT'"}
        },
        "required": ["ticker"]
    }
}

def get_company_info(ticker: str) -> str:
    try:
        info = _filter_info(yf.Ticker(ticker.upper().strip()).info)
        officers = info.get("companyOfficers") or []
        ceo = next(
            (o.get("name") for o in officers if "CEO" in (o.get("title") or "").upper()),
            officers[0].get("name") if officers else "N/A"
        )
        hq = ", ".join(filter(None, [info.get("city") or "", info.get("state") or ""]))
        desc = info.get("longBusinessSummary") or ""
        return json.dumps({
            "ticker": ticker.upper(),
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "employees": info.get("fullTimeEmployees"),
            "ceo": ceo,
            "hq": hq,
            "description": desc[:400] + ("…" if len(desc) > 400 else ""),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Financial Data — core metrics for all sectors ─────────────────────────────

financial_data_tool = {
    "name": "get_financial_data",
    "description": (
        "Core financial metrics applicable to all sectors: revenue growth, gross margin, "
        "FCF margin, Rule of 40, EV/Revenue, P/E, forward P/E, and analyst targets. "
        "For sector-specific profitability metrics use the sector-specific tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'NVDA'"}
        },
        "required": ["ticker"]
    }
}

def get_financial_data(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        info = _filter_info(stock.info)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not info or price is None:
            return json.dumps({"error": f"No data for '{ticker}'."})

        revenue_growth = None
        try:
            revs = stock.financials.loc["Total Revenue"].dropna()
            if len(revs) >= 2:
                revenue_growth = round(
                    ((revs.iloc[0] - revs.iloc[1]) / abs(revs.iloc[1])) * 100, 1
                )
        except Exception:
            pass

        fcf_margin = None
        try:
            fcf = stock.cashflow.loc["Free Cash Flow"].iloc[0]
            rev = info.get("totalRevenue") or 1
            fcf_margin = round((fcf / rev) * 100, 1)
        except Exception:
            pass

        rule_of_40 = (
            round(revenue_growth + fcf_margin, 1)
            if revenue_growth is not None and fcf_margin is not None
            else None
        )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "price": price,
            "market_cap": info.get("marketCap"),
            "pe_ratio": round(info.get("trailingPE") or 0, 1) or None,
            "forward_pe": round(info.get("forwardPE") or 0, 1) or None,
            "ev_revenue_multiple": round(info.get("enterpriseToRevenue") or 0, 1) or None,
            "ev_ebitda": round(info.get("enterpriseToEbitda") or 0, 1) or None,
            "revenue_growth_pct": revenue_growth,
            "gross_margin_pct": round((info.get("grossMargins") or 0) * 100, 1),
            "operating_margin_pct": round((info.get("operatingMargins") or 0) * 100, 1),
            "fcf_margin_pct": fcf_margin,
            "rule_of_40": rule_of_40,
            "short_interest_pct": round((info.get("shortPercentOfFloat") or 0) * 100, 1),
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_recommendation": (info.get("recommendationKey") or "N/A").replace("_", " "),
            "beta": info.get("beta"),
            "total_revenue": info.get("totalRevenue"),
            "data_source": "Yahoo Finance via yfinance",
        })
    except Exception as e:
        return json.dumps({"error": f"Financial data failed: {e}"})


# ── Macro Sensitivity ─────────────────────────────────────────────────────────
# Scores how sensitive the stock is to interest rates, USD strength, and GDP growth.
# Uses beta (market proxy), D/E ratio (rate sensitivity), and international revenue
# proxies to give portfolio managers a quick macro overlay for any ticker.

macro_sensitivity_tool = {
    "name": "get_macro_sensitivity",
    "description": (
        "Scores a stock's sensitivity to three macro factors: interest rates (via D/E ratio "
        "and dividend yield), USD strength (via international revenue proxy from business "
        "description and geographic segments), and GDP/economic cycle (via beta and "
        "sector cyclicality). Returns sensitivity ratings and plain-English explanations. "
        "Run this for every ticker — essential context for cross-asset portfolio decisions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'NVDA'"}
        },
        "required": ["ticker"]
    }
}

_INTL_KEYWORDS = [
    "international", "global", "worldwide", "europe", "asia", "china", "japan",
    "emea", "apac", "latin america", "rest of world", "non-us", "foreign",
]

def get_macro_sensitivity(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        info = _filter_info(stock.info)

        beta = info.get("beta")
        de_ratio = info.get("debtToEquity")           # percentage form in yfinance
        dividend_yield = info.get("dividendYield") or 0
        summary = (info.get("longBusinessSummary") or "").lower()

        # ── Rate sensitivity ───────────────────────────────────────────────────
        # High D/E + high dividend yield → high rate sensitivity
        if de_ratio is not None:
            de_norm = de_ratio / 100   # yfinance gives e.g. 42.5 meaning 0.425
        else:
            de_norm = None

        if de_norm is None:
            rate_score = "unknown"
            rate_reason = "Debt/equity data unavailable."
        elif de_norm > 2.0 or dividend_yield > 0.04:
            rate_score = "high"
            rate_reason = (
                f"D/E of {de_norm:.1f}× and dividend yield of {dividend_yield*100:.1f}% "
                "make this stock acutely sensitive to interest rate moves. "
                "Rising rates increase borrowing costs and compress yield-based valuations."
            )
        elif de_norm > 0.5 or dividend_yield > 0.02:
            rate_score = "moderate"
            rate_reason = (
                f"D/E of {de_norm:.1f}× provides some leverage but not enough for acute rate risk. "
                "Rate moves will affect the cost of refinancing but won't dominate returns."
            )
        else:
            rate_score = "low"
            rate_reason = (
                f"Low leverage (D/E {de_norm:.1f}×) and minimal dividend yield "
                "mean the stock is relatively insulated from rate-driven repricing."
            )

        # ── USD sensitivity ────────────────────────────────────────────────────
        intl_mentions = sum(1 for kw in _INTL_KEYWORDS if kw in summary)
        if intl_mentions >= 3:
            usd_score = "high"
            usd_reason = (
                "Business description contains multiple references to international operations. "
                "A strong USD creates a headwind to translated earnings from non-US geographies."
            )
        elif intl_mentions >= 1:
            usd_score = "moderate"
            usd_reason = (
                "Some international exposure mentioned. USD moves will partially affect revenue "
                "but the impact is likely manageable relative to domestic operations."
            )
        else:
            usd_score = "low"
            usd_reason = (
                "Business description suggests primarily domestic US operations. "
                "USD fluctuations are unlikely to materially impact reported results."
            )

        # ── GDP / Cycle sensitivity ─────────────────────────────────────────────
        sector = info.get("sector", "")
        cyclical_sectors = {
            "Consumer Cyclical", "Industrials", "Basic Materials",
            "Energy", "Real Estate", "Financial Services"
        }
        defensive_sectors = {"Consumer Defensive", "Healthcare", "Utilities"}

        if beta is None:
            gdp_score = "unknown"
            gdp_reason = "Beta data unavailable — GDP sensitivity cannot be estimated."
        elif beta > 1.5 or sector in cyclical_sectors:
            gdp_score = "high"
            gdp_reason = (
                f"Beta of {beta:.2f} and/or cyclical sector ({sector}) indicate strong "
                "correlation to economic activity. Earnings will compress meaningfully in a recession."
            )
        elif beta < 0.7 or sector in defensive_sectors:
            gdp_score = "low"
            gdp_reason = (
                f"Beta of {beta:.2f} and/or defensive sector ({sector}) suggest "
                "earnings are relatively stable across economic cycles."
            )
        else:
            gdp_score = "moderate"
            gdp_reason = (
                f"Beta of {beta:.2f} implies moderate correlation to the economic cycle. "
                "Earnings will be affected by a slowdown but not dramatically."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "sector": sector,
            "beta": beta,
            "debt_to_equity": round(de_norm, 2) if de_norm is not None else None,
            "dividend_yield_pct": round(dividend_yield * 100, 2),
            "sensitivity": {
                "interest_rates": {"score": rate_score, "explanation": rate_reason},
                "usd_strength":   {"score": usd_score,  "explanation": usd_reason},
                "gdp_cycle":      {"score": gdp_score,  "explanation": gdp_reason},
            },
            "data_source": "Yahoo Finance via yfinance",
        })
    except Exception as e:
        return json.dumps({"error": f"Macro sensitivity failed: {e}"})


# ── Insider Activity — SEC EDGAR Form 4 API ───────────────────────────────────
# Uses SEC EDGAR's free full-text search and submissions API.
# No scraping, no API key, no rate limits beyond respectful user-agent header.

insider_activity_tool = {
    "name": "get_insider_activity",
    "description": (
        "Fetch recent insider Form 4 filings from SEC EDGAR's free API "
        "(data.sec.gov/submissions). Returns the count of Form 4 filings "
        "in the last 90 days, filer names, and filing dates. "
        "Elevated filing activity during price weakness is a bullish signal. "
        "Works for any publicly listed US company."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'META'"}
        },
        "required": ["ticker"]
    }
}

def get_insider_activity(ticker: str) -> str:
    ticker = ticker.upper().strip()
    headers = {"User-Agent": "StockResearchTool research@example.com"}
    cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    try:
        # Step 1: Find company CIK via EDGAR company search
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&forms=4&dateRange=custom&startdt={cutoff}"
        )
        r = requests.get(search_url, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])

        if not hits:
            return json.dumps({
                "ticker": ticker,
                "filings_last_90_days": 0,
                "insider_signal": "neutral",
                "message": "No Form 4 filings found on SEC EDGAR in the last 90 days.",
                "data_source": "SEC EDGAR Form 4 Search (free API)",
            })

        filings = []
        for hit in hits[:25]:
            src = hit.get("_source", {})
            names = src.get("display_names") or []
            filings.append({
                "file_date": src.get("file_date", ""),
                "period": src.get("period_of_report", ""),
                "filer": names[0] if names else "Unknown",
                "accession": src.get("accession_no", ""),
            })

        # Recent = within cutoff window (already filtered by API)
        recent_count = len(filings)

        # Classify filing activity level
        if recent_count >= 10:
            signal = "elevated"
            signal_note = "High Form 4 activity — insiders are actively filing. Check individual filings for buy/sell direction."
        elif recent_count >= 4:
            signal = "moderate"
            signal_note = "Moderate insider filing activity in the past 90 days."
        elif recent_count >= 1:
            signal = "low"
            signal_note = "Light insider activity — only a few Form 4 filings in the past 90 days."
        else:
            signal = "none"
            signal_note = "No insider filings detected."

        return json.dumps({
            "ticker": ticker,
            "filings_last_90_days": recent_count,
            "insider_signal": signal,
            "signal_note": signal_note,
            "recent_filings": filings[:8],
            "data_source": "SEC EDGAR Form 4 Search (free API — data.sec.gov)",
            "note": (
                "Filing count reflects Form 4 activity. "
                "Buy vs. sell direction requires parsing individual EDGAR filing XML."
            ),
        })

    except Exception as e:
        return json.dumps({"error": f"SEC EDGAR insider fetch failed: {e}"})


# ── Sector Behavioral Biases ───────────────────────────────────────────────────
# Returns the top 3 retail misconceptions for the detected sector.
# No external API calls — uses the lookup table above.

sector_behavioral_biases_tool = {
    "name": "get_sector_behavioral_biases",
    "description": (
        "Returns the top 3 common retail investor behavioral biases for the detected sector. "
        "These frame the narrative vs. reality analysis — use them to construct the "
        "retail_narrative claim list in the synthesis. No API call required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sector": {
                "type": "string",
                "description": (
                    "Sector string from get_sector_profile, e.g. 'Technology', "
                    "'Healthcare', 'Financial Services', 'Consumer Cyclical', "
                    "'Energy', 'Real Estate', 'Industrials'."
                )
            }
        },
        "required": ["sector"]
    }
}

def get_sector_behavioral_biases(sector: str) -> str:
    biases = _SECTOR_BIASES.get(sector, _SECTOR_BIASES["_default"])
    return json.dumps({
        "sector": sector,
        "top_biases": biases,
        "usage_note": (
            "Use these biases to anchor the retail_narrative claims in the synthesis. "
            "Each bias should map to a specific claim that is then fact-checked against live data."
        ),
    })


# ── DCF Implied Growth ─────────────────────────────────────────────────────────
# Reverse-engineers the revenue CAGR embedded in the current stock price.
# Assumptions: 10% discount rate, 20× terminal FCF multiple, constant FCF margin.

dcf_implied_tool = {
    "name": "get_dcf_implied_growth",
    "description": (
        "Reverse-engineer the 10-year revenue CAGR the current stock price is implying. "
        "Uses 10% discount rate, 20× terminal FCF multiple, constant FCF margin. "
        "Returns implied CAGR, trailing 3-year actual CAGR, the gap, achievability verdict, "
        "and a plain-English summary. Run this for every ticker with positive FCF."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'NVDA'"}
        },
        "required": ["ticker"]
    }
}

def get_dcf_implied_growth(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = _filter_info(stock.info)

        price      = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap = info.get("marketCap")
        revenue    = info.get("totalRevenue")

        if not all([price, market_cap, revenue]) or revenue <= 0:
            return json.dumps({"error": f"Missing price/market cap/revenue for '{ticker}'."})

        fcf = None
        try:
            fcf = float(stock.cashflow.loc["Free Cash Flow"].iloc[0])
        except Exception:
            pass

        if fcf is None or fcf <= 0:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": (
                    "Cannot run DCF: free cash flow is negative or unavailable. "
                    "Pre-profitability companies cannot be valued on a DCF basis."
                ),
                "note": "Use EV/Revenue or EV/Gross Profit multiples instead.",
                "data_source": "Yahoo Finance via yfinance",
            })

        fcf_margin    = fcf / revenue
        discount_rate = 0.10
        terminal_mult = 20

        def dcf_value(cagr: float) -> float:
            x = (1 + cagr) / (1 + discount_rate)
            pv_flows = x * (1 - x**10) / (1 - x) if abs(x - 1) > 1e-9 else 10.0
            return revenue * fcf_margin * (pv_flows + terminal_mult * x**10)

        lo, hi = -0.30, 2.00
        for _ in range(80):
            mid = (lo + hi) / 2
            (lo if dcf_value(mid) < market_cap else hi).__class__  # silence lint
            if dcf_value(mid) < market_cap:
                lo = mid
            else:
                hi = mid
        implied_cagr = round(mid * 100, 1)

        trailing_cagr = None
        try:
            revs = stock.financials.loc["Total Revenue"].dropna()
            if len(revs) >= 4:
                trailing_cagr = round(
                    ((float(revs.iloc[0]) / float(revs.iloc[3])) ** (1/3) - 1) * 100, 1
                )
            elif len(revs) >= 2:
                n = len(revs) - 1
                trailing_cagr = round(
                    ((float(revs.iloc[0]) / float(revs.iloc[-1])) ** (1/n) - 1) * 100, 1
                )
        except Exception:
            pass

        gap = round(implied_cagr - trailing_cagr, 1) if trailing_cagr is not None else None

        if gap is None:
            verdict = "Insufficient historical data for comparison."
        elif gap <= 0:
            verdict = "Market prices in growth BELOW historical delivery — possible value opportunity."
        elif gap <= 5:
            verdict = "Market prices in growth roughly in line with history. Achievable if execution continues."
        elif gap <= 15:
            verdict = "Market prices in meaningfully above historical growth. Requires acceleration."
        else:
            verdict = "Market prices in exceptional growth far above historical rates. Requires a step-change."

        plain = (
            f"To justify ${price:.2f}/share, {info.get('longName', ticker)} "
            f"must grow revenue at {implied_cagr}% per year for 10 years. "
        )
        if trailing_cagr is not None:
            if gap is not None and abs(gap) <= 3:
                plain += f"History shows {trailing_cagr}% — roughly aligned with what the market assumes."
            elif gap is not None and gap > 0:
                plain += (
                    f"History shows {trailing_cagr}% — "
                    f"the market asks for {gap:.0f}pp more than it has delivered."
                )
            else:
                plain += (
                    f"History shows {trailing_cagr}% — "
                    f"the stock is priced for less growth than history suggests."
                )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "current_price": price,
            "market_cap": market_cap,
            "current_revenue": revenue,
            "current_fcf": round(fcf),
            "fcf_margin_pct": round(fcf_margin * 100, 1),
            "dcf_assumptions": {
                "discount_rate_pct": 10,
                "terminal_fcf_multiple": terminal_mult,
                "horizon_years": 10,
            },
            "implied_10yr_revenue_cagr_pct": implied_cagr,
            "trailing_3yr_revenue_cagr_pct": trailing_cagr,
            "gap_pct": gap,
            "verdict": verdict,
            "plain_english": plain,
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"DCF implied growth failed: {e}"})


# ── Dilution Rate ──────────────────────────────────────────────────────────────
# SBC is a real cost to shareholders that doesn't appear in GAAP earnings.
# This tool makes it explicit: SBC/revenue, SBC/GP, SBC-adjusted FCF margin,
# and YoY share count growth.

dilution_tool = {
    "name": "get_dilution_rate",
    "description": (
        "SBC as % of revenue/gross profit, SBC-adjusted FCF margin, and YoY share count growth. "
        "Benchmarks SBC level (low/moderate/high/very high) and flags >3% share dilution. "
        "Run this for every ticker — GAAP FCF overstates true cash when SBC is material."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'MSFT'"}
        },
        "required": ["ticker"]
    }
}

def get_dilution_rate(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = _filter_info(stock.info)
        cf     = stock.cashflow
        fin    = stock.financials
        result = {"ticker": ticker, "name": info.get("longName")}

        sbc = None
        for row_name in ["Stock Based Compensation",
                         "Share Based Compensation Expense",
                         "Stock Based Compensation Expense"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    sbc = float(s.iloc[0])
                break
        if sbc is None:
            result["sbc_error"] = "SBC not found in cash flow statement."

        revenue = info.get("totalRevenue")
        if not revenue:
            try:
                revs = fin.loc["Total Revenue"].dropna()
                revenue = float(revs.iloc[0]) if len(revs) >= 1 else None
            except Exception:
                pass

        gross_profit = None
        try:
            gp = fin.loc["Gross Profit"].dropna()
            gross_profit = float(gp.iloc[0]) if len(gp) >= 1 else None
        except Exception:
            pass

        fcf = None
        try:
            f = cf.loc["Free Cash Flow"].dropna()
            fcf = float(f.iloc[0]) if len(f) >= 1 else None
        except Exception:
            pass

        shares_current = info.get("sharesOutstanding")
        shares_prior   = None
        for row_name in ["Diluted Average Shares", "Ordinary Shares Number",
                         "Share Issued", "Basic Average Shares"]:
            try:
                for src in [fin, cf]:
                    if row_name in src.index:
                        s = src.loc[row_name].dropna()
                        if len(s) >= 2:
                            shares_prior = float(s.iloc[1])
                            break
                if shares_prior:
                    break
            except Exception:
                continue

        shares_growth = (
            round((shares_current - shares_prior) / abs(shares_prior) * 100, 2)
            if shares_current and shares_prior else None
        )

        sbc_pct_rev  = round(sbc / revenue * 100, 1)         if sbc and revenue       else None
        sbc_pct_gp   = round(sbc / gross_profit * 100, 1)    if sbc and gross_profit  else None
        gaap_fcf_mgn = round(fcf / revenue * 100, 1)         if fcf and revenue       else None
        sbc_adj_mgn  = round((fcf - sbc) / revenue * 100, 1) if fcf is not None and sbc and revenue else None
        overstate    = round(gaap_fcf_mgn - sbc_adj_mgn, 1)  if gaap_fcf_mgn and sbc_adj_mgn else None

        if sbc_pct_rev is not None:
            if sbc_pct_rev < 5:
                sbc_bench = f"Low ({sbc_pct_rev}% of revenue) — minimal dilution drag."
            elif sbc_pct_rev < 10:
                sbc_bench = f"Moderate ({sbc_pct_rev}% of revenue) — meaningful but manageable."
            elif sbc_pct_rev < 20:
                sbc_bench = f"High ({sbc_pct_rev}% of revenue) — significant drag; GAAP FCF overstated."
            else:
                sbc_bench = f"Very High ({sbc_pct_rev}% of revenue) — major dilution; true profitability substantially lower."
        else:
            sbc_bench = None

        if shares_growth is None:
            dilution_flag = "Share count history unavailable."
        elif shares_growth > 3:
            dilution_flag = (
                f"WARNING: Share count grew {shares_growth}% YoY — above the 3% warning threshold."
            )
        elif shares_growth > 0:
            dilution_flag = f"Share count grew {shares_growth}% YoY — modest dilution."
        else:
            dilution_flag = f"Share count declined {abs(shares_growth)}% YoY — net buyback activity."

        result.update({
            "shares_outstanding_current": shares_current,
            "shares_outstanding_prior_year": round(shares_prior) if shares_prior else None,
            "share_count_growth_pct": shares_growth,
            "dilution_flag": dilution_flag,
            "sbc_annual": round(sbc) if sbc else None,
            "sbc_pct_revenue": sbc_pct_rev,
            "sbc_pct_gross_profit": sbc_pct_gp,
            "sbc_benchmark": sbc_bench,
            "gaap_fcf_margin_pct": gaap_fcf_mgn,
            "sbc_adjusted_fcf_margin_pct": sbc_adj_mgn,
            "gaap_fcf_overstatement_ppt": overstate,
            "data_source": "Yahoo Finance via yfinance",
        })
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Dilution rate failed: {e}"})


# ── Exports ────────────────────────────────────────────────────────────────────

UNIVERSAL_TOOL_DEFS = [
    calculator_tool,
    sector_profile_tool,
    stock_price_tool,
    company_info_tool,
    financial_data_tool,
    macro_sensitivity_tool,
    insider_activity_tool,
    sector_behavioral_biases_tool,
    dcf_implied_tool,
    dilution_tool,
]

UNIVERSAL_FUNCTIONS = {
    "calculator": calculator,
    "get_sector_profile": get_sector_profile,
    "get_stock_price": get_stock_price,
    "get_company_info": get_company_info,
    "get_financial_data": get_financial_data,
    "get_macro_sensitivity": get_macro_sensitivity,
    "get_insider_activity": get_insider_activity,
    "get_sector_behavioral_biases": get_sector_behavioral_biases,
    "get_dcf_implied_growth": get_dcf_implied_growth,
    "get_dilution_rate": get_dilution_rate,
}
