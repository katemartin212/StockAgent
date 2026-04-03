#!/usr/bin/env python3
"""
tools_industrials.py
--------------------
Industrials sector tools:
  - get_book_to_bill       — backlog proxy via revenue acceleration
  - get_capacity_utilization — revenue-per-employee and capex intensity proxy
"""

import json
import yfinance as yf

from tools_base import _tool_schema

# ── Book-to-Bill ───────────────────────────────────────────────────────────────
# True B2B ratio = new orders / revenue, reported by semiconductors and industrials.
# This isn't available via free APIs for most companies.
# Proxy: quarterly revenue acceleration/deceleration as a backlog signal.
# Accelerating revenue → implied B2B > 1.0; decelerating → implied B2B < 1.0.

book_to_bill_tool = _tool_schema(
    "get_book_to_bill",
    "Estimates book-to-bill using quarterly revenue acceleration as a proxy. "
    "Accelerating sequential revenue growth implies orders exceed billings (B2B > 1). "
    "Decelerating implies backlog is shrinking (B2B < 1). Returns estimated B2B "
    "direction, revenue momentum, and 4-quarter quarterly revenue trend. "
    "True B2B requires earnings release order data; this tool uses financial proxies.",
    example="HON",
)

def get_book_to_bill(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials

        # Quarterly revenue — 6 quarters for trend
        quarterly_revs = []
        try:
            qrev = stock.quarterly_financials.loc["Total Revenue"].dropna()
            quarterly_revs = [float(v) for v in qrev.iloc[:6]]  # newest first
        except Exception:
            pass

        if len(quarterly_revs) < 3:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "Insufficient quarterly revenue data (need at least 3 quarters).",
                "data_source": "Yahoo Finance via yfinance",
            })

        # Sequential QoQ growth rates
        qoq_growth = []
        for i in range(len(quarterly_revs) - 1):
            prior = quarterly_revs[i + 1]
            if prior != 0:
                qoq_growth.append(round((quarterly_revs[i] - prior) / abs(prior) * 100, 1))

        # YoY for same-quarter comparison (cleaner for seasonal businesses)
        yoy_growth = []
        try:
            for i in range(min(4, len(quarterly_revs) - 4)):
                curr  = quarterly_revs[i]
                prior = quarterly_revs[i + 4]
                if prior != 0:
                    yoy_growth.append(round((curr - prior) / abs(prior) * 100, 1))
        except Exception:
            pass

        # Revenue acceleration/deceleration
        # Compare most recent QoQ growth to prior quarter QoQ growth
        recent_qoq  = qoq_growth[0] if len(qoq_growth) >= 1 else None
        prior_qoq   = qoq_growth[1] if len(qoq_growth) >= 2 else None
        acceleration = round(recent_qoq - prior_qoq, 1) if (recent_qoq is not None and prior_qoq is not None) else None

        # Annual revenue CAGR for context
        rev_cagr = None
        try:
            annual_revs = fin.loc["Total Revenue"].dropna()
            if len(annual_revs) >= 3:
                rev_cagr = round(
                    ((float(annual_revs.iloc[0]) / float(annual_revs.iloc[2])) ** 0.5 - 1) * 100, 1
                )
        except Exception:
            pass

        # B2B implied signal
        if acceleration is None:
            btb_signal = "unknown"
            btb_estimate = "insufficient data"
            assessment = "Cannot estimate B2B direction from available quarterly data."
        elif acceleration > 2:
            btb_signal = "above 1.0"
            btb_estimate = ">1.0 (implied)"
            assessment = (
                f"Revenue accelerating (+{acceleration}pp QoQ momentum improvement). "
                "Implies backlog is growing — orders likely exceeding billings. "
                "Positive leading indicator for future revenue."
            )
        elif acceleration < -2:
            btb_signal = "below 1.0"
            btb_estimate = "<1.0 (implied)"
            assessment = (
                f"Revenue decelerating ({acceleration}pp QoQ momentum decline). "
                "Implies backlog may be shrinking — orders trailing billings. "
                "Early warning signal for potential revenue softness ahead."
            )
        else:
            btb_signal = "near 1.0"
            btb_estimate = "~1.0 (implied)"
            assessment = (
                "Revenue momentum is roughly stable. "
                "Implies orders and billings are roughly matched — neutral B2B signal."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "quarterly_revenue_newest_first": [round(r) for r in quarterly_revs[:5]],
            "sequential_qoq_growth_pct": qoq_growth[:4],
            "yoy_growth_pct": yoy_growth,
            "revenue_acceleration_ppt": acceleration,
            "book_to_bill_estimate": btb_estimate,
            "btb_signal": btb_signal,
            "annual_revenue_cagr_pct": rev_cagr,
            "assessment": assessment,
            "data_caveat": (
                "True book-to-bill requires order intake data from earnings releases or "
                "semiconductor industry trackers (e.g. SEMI.org). "
                "This tool uses revenue acceleration as a financial proxy."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Book-to-bill analysis failed: {e}"})


# ── Capacity Utilization ───────────────────────────────────────────────────────
# True utilization requires production/capacity data from company disclosures.
# We proxy using: revenue per employee (throughput), capex/revenue trend (expansion signal),
# and operating leverage (whether revenue growth converts to operating income at a high rate).

capacity_util_tool = _tool_schema(
    "get_capacity_utilization",
    "Estimates capacity utilization using revenue per employee (throughput proxy), "
    "capex/revenue trend (capacity expansion signal), and operating leverage "
    "(how much of incremental revenue drops to operating income). "
    "Rising revenue-per-employee + declining capex/revenue = high utilization. "
    "Returns utilization signal, operating leverage measurement, and capex intensity.",
    example="CAT",
)

def get_capacity_utilization(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials
        cf     = stock.cashflow

        employees  = info.get("fullTimeEmployees")
        revenue    = info.get("totalRevenue")
        market_cap = info.get("marketCap")

        # Revenue per employee
        rev_per_emp = round(revenue / employees) if (revenue and employees) else None

        # Capex intensity (capex / revenue) — high and rising = expanding capacity
        capex = None
        for row_name in ["Capital Expenditure", "Purchase Of Property Plant And Equipment",
                         "Capital Expenditures"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    capex = abs(float(s.iloc[0]))
                break

        capex_pct = round(capex / revenue * 100, 1) if (capex and revenue) else None

        # Capex trend (are they investing more or less as % of revenue?)
        capex_trend = []
        try:
            rev_s = fin.loc["Total Revenue"].dropna()
            cap_s = cf.loc["Capital Expenditure"].dropna() if "Capital Expenditure" in cf.index else None
            if cap_s is not None:
                for i in range(min(3, len(rev_s), len(cap_s))):
                    r = float(rev_s.iloc[i])
                    c = abs(float(cap_s.iloc[i]))
                    if r != 0:
                        capex_trend.append(round(c / r * 100, 1))
        except Exception:
            pass

        # Operating leverage: % change in operating income / % change in revenue
        op_leverage = None
        try:
            rev_s2 = fin.loc["Total Revenue"].dropna()
            oi_s   = fin.loc["Operating Income"].dropna()
            if len(rev_s2) >= 2 and len(oi_s) >= 2:
                rev_chg = (float(rev_s2.iloc[0]) - float(rev_s2.iloc[1])) / abs(float(rev_s2.iloc[1]))
                oi_chg  = (float(oi_s.iloc[0]) - float(oi_s.iloc[1])) / abs(float(oi_s.iloc[1])) if float(oi_s.iloc[1]) != 0 else None
                if oi_chg is not None and rev_chg != 0:
                    op_leverage = round(oi_chg / rev_chg, 2)
        except Exception:
            pass

        # Utilization signal
        cap_trend_dir = "unknown"
        if len(capex_trend) >= 2:
            if capex_trend[0] > capex_trend[-1] + 1:
                cap_trend_dir = "expanding"
            elif capex_trend[0] < capex_trend[-1] - 1:
                cap_trend_dir = "moderating"
            else:
                cap_trend_dir = "stable"

        if op_leverage is None:
            util_signal = "unknown"
            util_note = "Operating leverage could not be calculated."
        elif op_leverage > 2:
            util_signal = "high"
            util_note = (
                f"Operating leverage of {op_leverage}× — strong volume absorption. "
                "Fixed costs are being spread across higher revenue, consistent with high utilization."
            )
        elif op_leverage > 1:
            util_signal = "moderate"
            util_note = (
                f"Operating leverage of {op_leverage}× — moderate utilization. "
                "Revenue growth is converting to operating income at a healthy but not peak rate."
            )
        else:
            util_signal = "low"
            util_note = (
                f"Operating leverage below 1× — cost growth is outpacing revenue growth. "
                "May indicate significant idle capacity, wage/materials inflation, or mix headwinds."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "full_time_employees": employees,
            "revenue_per_employee": rev_per_emp,
            "capex_annual": round(capex) if capex else None,
            "capex_pct_revenue": capex_pct,
            "capex_intensity_trend_pct": capex_trend,
            "capex_trend_direction": cap_trend_dir,
            "operating_leverage_ratio": op_leverage,
            "utilization_signal": util_signal,
            "utilization_note": util_note,
            "data_caveat": (
                "True capacity utilization (%) requires disclosed nameplate capacity data "
                "from company reports. Operating leverage and capex intensity are used as proxies."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Capacity utilization analysis failed: {e}"})


# ── Exports ────────────────────────────────────────────────────────────────────

INDUSTRIALS_TOOL_DEFS = [book_to_bill_tool, capacity_util_tool]
INDUSTRIALS_FUNCTIONS = {
    "get_book_to_bill": get_book_to_bill,
    "get_capacity_utilization": get_capacity_utilization,
}
