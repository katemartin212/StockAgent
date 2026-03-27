#!/usr/bin/env python3
"""
tools_consumer.py
-----------------
Consumer Cyclical / Consumer Defensive sector tools:
  - get_same_store_sales       — organic demand proxy via revenue-per-unit trends
  - get_inventory_turns        — COGS / average inventory; demand early warning
  - get_gross_margin_by_channel — overall GM trend + segment note
"""

import json
import yfinance as yf

# ── Same-Store Sales ───────────────────────────────────────────────────────────
# True SSS (comps) are not reported via free APIs — they're disclosed quarterly
# in earnings releases and SEC filings. We proxy using revenue-per-employee
# as a throughput proxy, plus YoY quarterly revenue trends to strip out some
# of the new-store contribution noise.

sss_tool = {
    "name": "get_same_store_sales",
    "description": (
        "Estimates same-store / comparable sales growth using revenue-per-employee as a "
        "throughput proxy, quarterly revenue YoY growth trends, and gross margin trajectory. "
        "True comparable sales data requires earnings release parsing. Returns a proxy estimate "
        "with trend direction and what it implies for organic demand."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'HD'"}
        },
        "required": ["ticker"]
    }
}

def get_same_store_sales(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info

        employees = info.get("fullTimeEmployees")
        revenue   = info.get("totalRevenue")

        # Revenue per employee — proxy for store/unit productivity
        rev_per_employee = round(revenue / employees) if (revenue and employees) else None

        # Quarterly revenue YoY growth — gives organic demand signal net of some noise
        quarterly_yoy = []
        try:
            qrev = stock.quarterly_financials.loc["Total Revenue"].dropna()
            for i in range(min(4, len(qrev) - 4)):
                curr  = float(qrev.iloc[i])
                prior = float(qrev.iloc[i + 4])
                if prior != 0:
                    quarterly_yoy.append(round((curr - prior) / abs(prior) * 100, 1))
        except Exception:
            pass

        # Gross margin trend (margin compression often precedes SSS decline)
        gm_trend = []
        try:
            fin = stock.financials
            gm_series  = fin.loc["Gross Profit"].dropna()
            rev_series = fin.loc["Total Revenue"].dropna()
            for i in range(min(3, len(gm_series), len(rev_series))):
                gm_pct = float(gm_series.iloc[i]) / float(rev_series.iloc[i]) * 100
                gm_trend.append(round(gm_pct, 1))
        except Exception:
            pass

        # Proxy SSS: average of quarterly YoY growth rates
        proxy_sss = round(sum(quarterly_yoy) / len(quarterly_yoy), 1) if quarterly_yoy else None

        # Trend assessment
        if not quarterly_yoy:
            organic_trend = "insufficient data"
            interpretation = "Quarterly revenue data unavailable."
        elif proxy_sss is not None and proxy_sss > 5:
            organic_trend = "strong"
            interpretation = (
                f"Quarterly revenue growing ~{proxy_sss}% YoY — consistent with positive comp trends. "
                "Organic demand appears robust."
            )
        elif proxy_sss is not None and proxy_sss > 0:
            organic_trend = "positive"
            interpretation = (
                f"Quarterly revenue growing ~{proxy_sss}% YoY — modest positive organic demand. "
                "Watch for margin signals that distinguish pricing power from volume."
            )
        elif proxy_sss is not None and proxy_sss > -3:
            organic_trend = "flat"
            interpretation = (
                f"Quarterly revenue flat to slightly negative ({proxy_sss}% YoY). "
                "Comps are likely flat or slightly negative once new openings are stripped out."
            )
        else:
            organic_trend = "negative"
            interpretation = (
                f"Quarterly revenue declining ~{proxy_sss}% YoY — likely negative comp performance. "
                "Demand headwinds may be building faster than new unit openings can offset."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "full_time_employees": employees,
            "revenue_per_employee": rev_per_employee,
            "quarterly_yoy_revenue_growth_pct": quarterly_yoy,
            "proxy_sss_growth_pct": proxy_sss,
            "organic_demand_trend": organic_trend,
            "gross_margin_by_year_pct": gm_trend,
            "interpretation": interpretation,
            "data_caveat": (
                "True same-store sales require parsing earnings releases. "
                "This tool uses revenue trends and margin signals as proxies."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Same-store sales proxy failed: {e}"})


# ── Inventory Turns ────────────────────────────────────────────────────────────
# Inventory Turns = COGS / Average Inventory
# This is one of the clearest leading indicators of demand weakness in retail.
# Turns decline 1–2 quarters before markdowns show up in reported gross margins.

inventory_turns_tool = {
    "name": "get_inventory_turns",
    "description": (
        "Inventory Turns = COGS / Average Inventory. Declining turns signal demand "
        "weakness before it appears in revenue or gross margins (typically 1–2 quarters lead). "
        "Returns trailing turns, trend over 3 years, days inventory outstanding, and "
        "a demand signal. For retailers/consumer companies this is a critical leading indicator."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'WMT'"}
        },
        "required": ["ticker"]
    }
}

def get_inventory_turns(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials
        bs     = stock.balance_sheet

        # COGS
        cogs = None
        for row_name in ["Cost Of Revenue", "Cost Of Goods Sold", "Cost Of Goods And Services Sold"]:
            if row_name in fin.index:
                cogs_series = fin.loc[row_name].dropna()
                if len(cogs_series) >= 1:
                    cogs = abs(float(cogs_series.iloc[0]))
                break

        if cogs is None:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "COGS not found. Company may be a service business where inventory turns are not applicable.",
                "data_source": "Yahoo Finance via yfinance",
            })

        # Inventory from balance sheet
        inventory_current = None
        inventory_prior   = None
        for row_name in ["Inventory", "Inventories", "Raw Materials And Work In Process"]:
            if row_name in bs.index:
                inv_series = bs.loc[row_name].dropna()
                if len(inv_series) >= 1:
                    inventory_current = float(inv_series.iloc[0])
                if len(inv_series) >= 2:
                    inventory_prior = float(inv_series.iloc[1])
                break

        if inventory_current is None:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "Inventory data not found. May be a service/digital business.",
                "data_source": "Yahoo Finance via yfinance",
            })

        avg_inventory = (
            (inventory_current + inventory_prior) / 2
            if inventory_prior else inventory_current
        )

        turns = round(cogs / avg_inventory, 2) if avg_inventory else None
        dio   = round(365 / turns, 0) if turns else None  # Days Inventory Outstanding

        # Multi-year trend
        turns_by_year = []
        try:
            cogs_s = (
                fin.loc["Cost Of Revenue"].dropna()
                if "Cost Of Revenue" in fin.index
                else fin.loc["Cost Of Goods Sold"].dropna()
                if "Cost Of Goods Sold" in fin.index else None
            )
            inv_s = (
                bs.loc["Inventory"].dropna()
                if "Inventory" in bs.index
                else bs.loc["Inventories"].dropna()
                if "Inventories" in bs.index else None
            )
            if cogs_s is not None and inv_s is not None:
                for i in range(min(len(cogs_s), len(inv_s), 3)):
                    c = abs(float(cogs_s.iloc[i]))
                    iv = float(inv_s.iloc[i])
                    if iv != 0:
                        turns_by_year.append(round(c / iv, 2))
        except Exception:
            pass

        # Trend
        trend = "unknown"
        if len(turns_by_year) >= 2:
            if turns_by_year[0] > turns_by_year[-1] * 1.05:
                trend = "improving"
            elif turns_by_year[0] < turns_by_year[-1] * 0.95:
                trend = "deteriorating"
            else:
                trend = "stable"

        # Signal
        if trend == "deteriorating":
            signal = "bearish"
            signal_note = (
                "Declining inventory turns signal demand is weakening relative to supply. "
                "Expect markdown pressure and gross margin compression in 1–2 quarters."
            )
        elif trend == "improving":
            signal = "bullish"
            signal_note = (
                "Improving turns indicate demand is outpacing inventory build. "
                "Supports pricing power and margin stability."
            )
        else:
            signal = "neutral"
            signal_note = "Stable inventory turns — no clear demand acceleration or deterioration signal."

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "cogs_annual": round(cogs),
            "inventory_current": round(inventory_current),
            "average_inventory": round(avg_inventory),
            "inventory_turns": turns,
            "days_inventory_outstanding": int(dio) if dio else None,
            "turns_by_year": turns_by_year,
            "turns_trend": trend,
            "demand_signal": signal,
            "signal_note": signal_note,
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Inventory turns calculation failed: {e}"})


# ── Gross Margin by Channel ────────────────────────────────────────────────────
# True channel-level margin requires segment disclosures from 10-K.
# We return overall gross margin trajectory and flag the limitation clearly,
# plus any segment revenue data that yfinance exposes.

gm_channel_tool = {
    "name": "get_gross_margin_by_channel",
    "description": (
        "Returns overall gross margin trend over 3 years and notes any segment-level "
        "data available via yfinance. True channel-level margin breakdown (digital vs "
        "physical, online vs in-store) requires 10-K segment disclosures. "
        "Returns a margin trajectory assessment and notes the data limitation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'TGT'"}
        },
        "required": ["ticker"]
    }
}

def get_gross_margin_by_channel(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials

        gm_by_year = []
        try:
            gm_series  = fin.loc["Gross Profit"].dropna()
            rev_series = fin.loc["Total Revenue"].dropna()
            for i in range(min(4, len(gm_series), len(rev_series))):
                gm = float(gm_series.iloc[i])
                rv = float(rev_series.iloc[i])
                if rv != 0:
                    gm_by_year.append(round(gm / rv * 100, 1))
        except Exception:
            pass

        current_gm = gm_by_year[0] if gm_by_year else None

        # Overall trend
        trend = "unknown"
        trend_note = "Insufficient data for trend analysis."
        if len(gm_by_year) >= 3:
            if gm_by_year[0] > gm_by_year[-1] + 2:
                trend = "expanding"
                trend_note = (
                    f"Gross margin has expanded from {gm_by_year[-1]}% to {gm_by_year[0]}% — "
                    "may reflect mix shift toward higher-margin channels or pricing power."
                )
            elif gm_by_year[0] < gm_by_year[-1] - 2:
                trend = "compressing"
                trend_note = (
                    f"Gross margin compressed from {gm_by_year[-1]}% to {gm_by_year[0]}% — "
                    "may reflect channel mix shift toward lower-margin formats, markdowns, or cost inflation."
                )
            else:
                trend = "stable"
                trend_note = f"Gross margin stable at approximately {current_gm}%."

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "gross_margin_current_pct": current_gm,
            "gross_margin_by_year_pct": gm_by_year,
            "gross_margin_trend": trend,
            "trend_analysis": trend_note,
            "data_caveat": (
                "Channel-level margin breakdown (digital vs in-store, direct vs wholesale) "
                "requires segment disclosures from 10-K annual reports. "
                "This tool returns overall gross margin trajectory as the available proxy."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Gross margin by channel failed: {e}"})


# ── Exports ────────────────────────────────────────────────────────────────────

CONSUMER_TOOL_DEFS = [sss_tool, inventory_turns_tool, gm_channel_tool]
CONSUMER_FUNCTIONS = {
    "get_same_store_sales": get_same_store_sales,
    "get_inventory_turns": get_inventory_turns,
    "get_gross_margin_by_channel": get_gross_margin_by_channel,
}
