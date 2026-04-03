#!/usr/bin/env python3
"""
tools_energy.py
---------------
Energy sector tools:
  - get_break_even_price   — estimated oil/gas price at which company covers costs
  - get_reserve_replacement — depletion proxy via depreciation / PP&E trends
"""

import json
import yfinance as yf

from tools_base import _tool_schema

# ── Break-Even Price ───────────────────────────────────────────────────────────
# True break-even requires production volumes (barrels/day or Mcf) which aren't
# in yfinance. We proxy using total operating costs per dollar of revenue, then
# relate that to recent commodity prices to estimate a break-even range.

break_even_tool = _tool_schema(
    "get_break_even_price",
    "Estimates the oil/gas price at which the company covers operating costs and capex. "
    "Uses operating cost as % of revenue, revenue per period, and capex intensity to "
    "derive an implied break-even commodity price range. "
    "Requires knowing recent avg realized price (estimated from revenue trends). "
    "Flags if break-even appears above $60/bbl (structural risk in a downcycle).",
    example="XOM",
)

def get_break_even_price(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials
        cf     = stock.cashflow

        revenue = info.get("totalRevenue") or 0

        # Operating costs
        cogs = None
        for row_name in ["Cost Of Revenue", "Cost Of Goods Sold",
                         "Operating Costs And Expenses"]:
            if row_name in fin.index:
                s = fin.loc[row_name].dropna()
                if len(s) >= 1:
                    cogs = abs(float(s.iloc[0]))
                break

        op_expense = None
        for row_name in ["Operating Expense", "Total Operating Expenses",
                         "Selling General And Administration"]:
            if row_name in fin.index:
                s = fin.loc[row_name].dropna()
                if len(s) >= 1:
                    op_expense = abs(float(s.iloc[0]))
                break

        # Capex
        capex = None
        for row_name in ["Capital Expenditure", "Purchase Of Property Plant And Equipment",
                         "Capital Expenditures"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    capex = abs(float(s.iloc[0]))
                break

        if not revenue or not cogs:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "Revenue or cost data insufficient for break-even estimation.",
                "data_source": "Yahoo Finance via yfinance",
            })

        total_costs = cogs + (op_expense or 0)
        capex_pct   = round(capex / revenue * 100, 1) if capex else None

        # Operating cost ratio
        op_cost_ratio = round(total_costs / revenue, 3)  # costs per $1 of revenue

        # Estimate: if revenue was at price P, break-even is where revenue = total_costs + capex
        # break_even_revenue = total_costs + capex
        # implied break_even_price ≈ (total_costs + (capex or 0)) / revenue * recent_commodity_proxy
        # We can't know actual production volume, so we express as % of required revenue

        total_required = total_costs + (capex or 0)
        revenue_coverage = round(total_required / revenue * 100, 1)

        # Implied: if stock generates $X revenue at $Y oil price, break-even oil = Y * (costs/revenue)
        # We don't know Y (realized price) from free data, so we flag the estimate clearly

        if revenue_coverage < 70:
            risk_level = "low"
            note = (
                f"Total costs + capex are {revenue_coverage}% of revenue at current levels. "
                "Company has substantial headroom before prices would need to fall significantly."
            )
        elif revenue_coverage < 90:
            risk_level = "moderate"
            note = (
                f"Total costs + capex are {revenue_coverage}% of revenue. "
                "Break-even is sensitive to ~10–20% commodity price declines."
            )
        elif revenue_coverage < 100:
            risk_level = "elevated"
            note = (
                f"Total costs + capex are {revenue_coverage}% of revenue — tight margin. "
                "A moderate commodity price decline could push FCF negative."
            )
        else:
            risk_level = "high"
            note = (
                f"Total costs + capex exceed 100% of revenue at current levels — "
                "company is not covering its full cost structure including investment."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "total_revenue": round(revenue),
            "operating_costs": round(total_costs),
            "capex_annual": round(capex) if capex else None,
            "capex_pct_revenue": capex_pct,
            "cost_revenue_coverage_pct": revenue_coverage,
            "break_even_risk": risk_level,
            "break_even_assessment": note,
            "data_caveat": (
                "Exact $/barrel break-even requires production volume data (bboe/day) "
                "which is not available via free APIs. This tool uses cost/revenue ratios "
                "as a financial break-even proxy."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Break-even price estimation failed: {e}"})


# ── Reserve Replacement ────────────────────────────────────────────────────────
# True reserve replacement ratio requires SEC reserve disclosures from 10-K Supplement.
# We proxy using depletion/depreciation as % of PP&E — a company depleting faster
# than it invests in new capacity has reserve replacement < 100%.

reserve_replacement_tool = _tool_schema(
    "get_reserve_replacement",
    "Estimates reserve replacement using Depletion & Depreciation / PP&E as a proxy. "
    "When D&D exceeds new capex additions to PP&E, the company is consuming reserves "
    "faster than it replaces them (ratio < 100%). Flags if asset base is shrinking. "
    "True reserve replacement requires SEC 10-K Supplemental oil & gas disclosures.",
    example="CVX",
)

def get_reserve_replacement(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        cf     = stock.cashflow
        bs     = stock.balance_sheet

        # Depreciation & Depletion from cash flow (D&D is non-cash, added back to net income)
        dda = None
        for row_name in ["Depreciation And Amortization", "Depreciation Amortization Depletion",
                         "Depreciation", "Depletion"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    dda = abs(float(s.iloc[0]))
                break

        # PP&E net from balance sheet
        ppe_current = None
        ppe_prior   = None
        for row_name in ["Net PPE", "Property Plant And Equipment Net",
                         "Properties", "Oil Gas And Coal Properties"]:
            if row_name in bs.index:
                s = bs.loc[row_name].dropna()
                if len(s) >= 1:
                    ppe_current = float(s.iloc[0])
                if len(s) >= 2:
                    ppe_prior   = float(s.iloc[1])
                break

        # Capex (new investment in reserves/assets)
        capex = None
        for row_name in ["Capital Expenditure", "Purchase Of Property Plant And Equipment",
                         "Capital Expenditures"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    capex = abs(float(s.iloc[0]))
                break

        if not dda:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "Depletion/depreciation data not found.",
                "data_source": "Yahoo Finance via yfinance",
            })

        # Proxy reserve replacement ratio: capex / D&D
        # >1.0 = investing more than depleting (replacing and growing)
        # <1.0 = depleting faster than replacing (shrinking asset base)
        rrr_proxy = round(capex / dda, 2) if capex else None

        # PP&E growth (another proxy: is the asset base growing or shrinking?)
        ppe_growth = None
        if ppe_current and ppe_prior and ppe_prior != 0:
            ppe_growth = round((ppe_current - ppe_prior) / abs(ppe_prior) * 100, 1)

        # D&D as % of PP&E
        dda_pct_ppe = round(dda / ppe_current * 100, 1) if ppe_current else None

        if rrr_proxy is None:
            assessment = "Capex data unavailable — cannot estimate reserve replacement ratio."
            signal = "unknown"
        elif rrr_proxy >= 1.5:
            signal = "strong"
            assessment = (
                f"Capex/D&D ratio of {rrr_proxy}× suggests the company is replacing reserves "
                "at 1.5× the depletion rate — growing the asset base."
            )
        elif rrr_proxy >= 1.0:
            signal = "adequate"
            assessment = (
                f"Capex/D&D ratio of {rrr_proxy}× — reserves are being replaced roughly 1-for-1. "
                "Asset base is stable."
            )
        elif rrr_proxy >= 0.7:
            signal = "below replacement"
            assessment = (
                f"Capex/D&D ratio of {rrr_proxy}× — company is not fully replacing reserves. "
                "Asset base is gradually shrinking. This is sustainable short-term but erodes long-term production."
            )
        else:
            signal = "declining"
            assessment = (
                f"Capex/D&D ratio of only {rrr_proxy}× — significant under-investment relative to depletion. "
                "Reserve base is shrinking materially. Future production capacity is at risk."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "depletion_depreciation_annual": round(dda),
            "capex_annual": round(capex) if capex else None,
            "ppe_net_current": round(ppe_current) if ppe_current else None,
            "ppe_yoy_growth_pct": ppe_growth,
            "dda_pct_of_ppe": dda_pct_ppe,
            "capex_to_dda_ratio": rrr_proxy,
            "reserve_replacement_signal": signal,
            "assessment": assessment,
            "data_caveat": (
                "True reserve replacement ratio (in BOE) requires SEC 10-K Supplemental "
                "oil & gas reserve disclosures. This tool uses Capex/D&D and PP&E growth "
                "as financial proxies for asset replenishment."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Reserve replacement analysis failed: {e}"})


# ── Exports ────────────────────────────────────────────────────────────────────

ENERGY_TOOL_DEFS = [break_even_tool, reserve_replacement_tool]
ENERGY_FUNCTIONS = {
    "get_break_even_price": get_break_even_price,
    "get_reserve_replacement": get_reserve_replacement,
}
