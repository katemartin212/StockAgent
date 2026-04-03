#!/usr/bin/env python3
"""
tools_realestate.py
-------------------
Real Estate / REIT sector tools:
  - get_ffo     — Funds From Operations (the correct REIT profitability metric)
  - get_cap_rate — implied cap rate from NOI / total asset value
"""

import json
import yfinance as yf

from tools_base import _tool_schema

# ── Funds From Operations ──────────────────────────────────────────────────────
# FFO = Net Income + Depreciation & Amortization - Gains on Property Sales
# P/FFO is the correct valuation multiple for REITs, not P/E.
# GAAP net income includes large depreciation charges that make earnings
# meaningless for real property — FFO adds them back.

ffo_tool = _tool_schema(
    "get_ffo",
    "Calculates Funds From Operations (FFO) = Net Income + D&A - Gains on Property Sales. "
    "FFO is the standard REIT profitability metric. Returns FFO per share, P/FFO multiple, "
    "FFO yield, and assessment. P/FFO below 15× is generally undervalued for quality REITs; "
    "above 25× is expensive. GAAP P/E is meaningless for REITs due to depreciation.",
    example="SPG",
)

def get_ffo(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials
        cf     = stock.cashflow

        price  = info.get("currentPrice") or info.get("regularMarketPrice")
        shares = info.get("sharesOutstanding")

        # Net income
        net_income = None
        for row_name in ["Net Income", "Net Income Common Stockholders",
                         "Net Income Including Noncontrolling Interests"]:
            if row_name in fin.index:
                s = fin.loc[row_name].dropna()
                if len(s) >= 1:
                    net_income = float(s.iloc[0])
                break

        # D&A from cash flow statement (non-cash add-back)
        dna = None
        for row_name in ["Depreciation And Amortization", "Depreciation Amortization Depletion",
                         "Depreciation"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    dna = abs(float(s.iloc[0]))
                break

        # Gains on property sales (subtract from FFO)
        # Often captured in "Gain Loss On Sale Of Ppe" or similar
        gains_on_sales = 0
        for row_name in ["Gain Loss On Sale Of Ppe", "Gain On Sale Of Business",
                         "Net Gain Loss On Investments"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    gains_on_sales = float(s.iloc[0])
                break

        if net_income is None or dna is None:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "Net income or D&A data unavailable for FFO calculation.",
                "data_source": "Yahoo Finance via yfinance",
            })

        ffo = net_income + dna - gains_on_sales
        ffo_per_share = round(ffo / shares, 2) if shares else None
        p_ffo = round(price / ffo_per_share, 1) if (price and ffo_per_share and ffo_per_share > 0) else None
        ffo_yield = round(ffo_per_share / price * 100, 2) if (price and ffo_per_share and price > 0) else None

        # Year-over-year FFO growth (need prior year net income + D&A)
        ffo_prior = None
        ffo_growth = None
        try:
            ni_s   = fin.loc["Net Income"].dropna()
            dna_s  = cf.loc["Depreciation And Amortization"].dropna()
            if len(ni_s) >= 2 and len(dna_s) >= 2:
                ni_prior  = float(ni_s.iloc[1])
                dna_prior = abs(float(dna_s.iloc[1]))
                ffo_prior = ni_prior + dna_prior
                if ffo_prior != 0:
                    ffo_growth = round((ffo - ffo_prior) / abs(ffo_prior) * 100, 1)
        except Exception:
            pass

        # Assessment
        if p_ffo is None:
            assessment = "P/FFO could not be calculated — share count or price data missing."
        elif p_ffo < 12:
            assessment = f"P/FFO of {p_ffo}× is below 12× — potentially undervalued or pricing in stress."
        elif p_ffo < 18:
            assessment = f"P/FFO of {p_ffo}× is in the fair value range for quality REITs (12–18×)."
        elif p_ffo < 25:
            assessment = f"P/FFO of {p_ffo}× is toward the upper range — pricing in growth or premium assets."
        else:
            assessment = f"P/FFO of {p_ffo}× is above 25× — elevated; requires exceptional growth to justify."

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "current_price": price,
            "net_income": round(net_income),
            "depreciation_amortization": round(dna),
            "gains_on_property_sales": round(gains_on_sales),
            "ffo": round(ffo),
            "ffo_per_share": ffo_per_share,
            "ffo_prior_year": round(ffo_prior) if ffo_prior else None,
            "ffo_yoy_growth_pct": ffo_growth,
            "p_ffo_multiple": p_ffo,
            "ffo_yield_pct": ffo_yield,
            "assessment": assessment,
            "data_note": (
                "FFO = Net Income + D&A - Gains on Property Sales. "
                "NAREIT-defined AFFO (Adjusted FFO) further deducts recurring capex — "
                "available only from company earnings supplements."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"FFO calculation failed: {e}"})


# ── Cap Rate ───────────────────────────────────────────────────────────────────
# Cap Rate = Net Operating Income / Property Value
# NOI = Revenue - Operating Expenses (before debt service, depreciation)
# Property Value is approximated from total assets

cap_rate_tool = _tool_schema(
    "get_cap_rate",
    "Estimates implied cap rate = Net Operating Income / Total Property Value. "
    "NOI proxied from operating income + D&A (adding back non-cash charges). "
    "Property value uses total assets from balance sheet. "
    "Cap rates below 4% are expensive (low yield); above 7% is high yield/higher risk. "
    "Returns cap rate, NOI, and assessment vs. current rate environment.",
    example="O",
)

def get_cap_rate(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials
        cf     = stock.cashflow
        bs     = stock.balance_sheet

        # Operating income
        op_income = None
        for row_name in ["Operating Income", "Total Operating Income As Reported",
                         "Ebit"]:
            if row_name in fin.index:
                s = fin.loc[row_name].dropna()
                if len(s) >= 1:
                    op_income = float(s.iloc[0])
                break

        # D&A (add back to get NOI from operating income)
        dna = None
        for row_name in ["Depreciation And Amortization", "Depreciation Amortization Depletion"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    dna = abs(float(s.iloc[0]))
                break

        # Total assets as property value proxy
        total_assets = None
        for row_name in ["Total Assets", "Total Assets Net Minority Interest"]:
            if row_name in bs.index:
                s = bs.loc[row_name].dropna()
                if len(s) >= 1:
                    total_assets = float(s.iloc[0])
                break

        if op_income is None:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "Operating income not found — cannot calculate NOI.",
                "data_source": "Yahoo Finance via yfinance",
            })

        # NOI = Operating Income + D&A (add back non-cash depreciation to approximate cash NOI)
        noi = op_income + (dna or 0)
        cap_rate = round(noi / total_assets * 100, 2) if total_assets else None

        # Market cap implied cap rate (on equity value)
        market_cap = info.get("marketCap")
        total_debt = info.get("totalDebt") or 0
        ev = (market_cap or 0) + total_debt
        ev_cap_rate = round(noi / ev * 100, 2) if ev > 0 else None

        if cap_rate is None:
            assessment = "Cap rate could not be calculated — property asset data unavailable."
        elif cap_rate < 4:
            assessment = (
                f"Implied cap rate of {cap_rate}% is below 4% — premium valuation. "
                "Reflects high-quality assets or significant cap rate compression in current environment."
            )
        elif cap_rate < 6:
            assessment = (
                f"Cap rate of {cap_rate}% is in the core real estate range (4–6%). "
                "Typical for high-quality commercial properties in major markets."
            )
        elif cap_rate < 8:
            assessment = (
                f"Cap rate of {cap_rate}% is value/core-plus territory. "
                "Higher yield reflects either value-add opportunity or secondary market assets."
            )
        else:
            assessment = (
                f"Cap rate of {cap_rate}% is above 8% — high yield territory. "
                "May reflect distress, lower-quality assets, or market dislocation."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "operating_income": round(op_income),
            "depreciation_added_back": round(dna) if dna else None,
            "noi_estimate": round(noi),
            "total_assets": round(total_assets) if total_assets else None,
            "implied_cap_rate_pct": cap_rate,
            "ev_cap_rate_pct": ev_cap_rate,
            "assessment": assessment,
            "data_caveat": (
                "NOI is approximated from operating income + D&A. "
                "True NOI excludes management fees and one-time items available only in property supplements. "
                "Total assets is used as a proxy for property value — book value may diverge from market value."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Cap rate calculation failed: {e}"})


# ── Exports ────────────────────────────────────────────────────────────────────

REALESTATE_TOOL_DEFS = [ffo_tool, cap_rate_tool]
REALESTATE_FUNCTIONS = {
    "get_ffo": get_ffo,
    "get_cap_rate": get_cap_rate,
}
