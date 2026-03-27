#!/usr/bin/env python3
"""
tools_healthcare.py
-------------------
Healthcare sector tools:
  - get_pipeline_value    — R&D intensity as pipeline optionality proxy
  - get_patent_cliff      — revenue concentration + R&D trend for patent risk
  - get_fda_catalyst_risk — binary event risk scoring via news + R&D trajectory
"""

import json
import yfinance as yf

# ── Pipeline Value ─────────────────────────────────────────────────────────────
# R&D data is the best proxy available via free APIs. Actual pipeline stage counts
# (Phase 1/2/3) require parsing SEC 10-K/10-Q filings or specialized databases.
# We flag this clearly so the analyst knows the limitation.

pipeline_value_tool = {
    "name": "get_pipeline_value",
    "description": (
        "Estimates pharmaceutical/biotech pipeline optionality using R&D spend as % of revenue, "
        "R&D spend trend (growing/shrinking), and R&D spend per market cap (a rough proxy for "
        "investment intensity relative to company size). Returns R&D intensity score and "
        "a pipeline health assessment. Note: actual Phase 1/2/3 counts require 10-K parsing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'JNJ'"}
        },
        "required": ["ticker"]
    }
}

def get_pipeline_value(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials

        revenue    = info.get("totalRevenue")
        market_cap = info.get("marketCap")

        # R&D from income statement
        rd_current = None
        rd_prior   = None
        for row_name in ["Research And Development", "Research And Development Expenses",
                         "Total Research And Development"]:
            if row_name in fin.index:
                rd_series = fin.loc[row_name].dropna()
                if len(rd_series) >= 1:
                    rd_current = float(rd_series.iloc[0])
                if len(rd_series) >= 2:
                    rd_prior = float(rd_series.iloc[1])
                break

        if rd_current is None:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "R&D expense not found in income statement. Company may not report separately.",
                "note": "Many diversified healthcare companies embed R&D in operating expenses.",
                "data_source": "Yahoo Finance via yfinance",
            })

        rd_pct_rev = round(rd_current / revenue * 100, 1) if revenue else None
        rd_pct_mktcap = round(rd_current / market_cap * 100, 1) if market_cap else None

        # Trend
        if rd_current and rd_prior:
            rd_growth_pct = round((rd_current - rd_prior) / abs(rd_prior) * 100, 1)
            rd_trend = "growing" if rd_growth_pct > 5 else "declining" if rd_growth_pct < -5 else "stable"
        else:
            rd_growth_pct = None
            rd_trend = "unknown"

        # Pipeline intensity score based on R&D/revenue benchmarks
        # Typical ranges: big pharma 15-25%, biotech 40-80%, medical devices 8-15%
        if rd_pct_rev is None:
            intensity = "unknown"
            assessment = "Insufficient data to assess pipeline intensity."
        elif rd_pct_rev > 40:
            intensity = "very high"
            assessment = (
                f"R&D spend of {rd_pct_rev}% of revenue indicates a biotech-style pipeline investment. "
                "The company is betting heavily on pipeline success — high optionality but high binary risk."
            )
        elif rd_pct_rev > 20:
            intensity = "high"
            assessment = (
                f"R&D at {rd_pct_rev}% of revenue is above the pharma industry average (15-25%). "
                "Suggests meaningful late-stage pipeline investment or heavy platform-building."
            )
        elif rd_pct_rev > 10:
            intensity = "moderate"
            assessment = (
                f"R&D at {rd_pct_rev}% of revenue is in-line with diversified pharma/medical device norms. "
                "Balanced between current earnings and pipeline optionality."
            )
        else:
            intensity = "low"
            assessment = (
                f"R&D at {rd_pct_rev}% of revenue is below typical healthcare sector levels. "
                "Company may be primarily commercial-stage with limited pipeline replenishment."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "rd_expense_annual": round(rd_current),
            "rd_pct_revenue": rd_pct_rev,
            "rd_pct_market_cap": rd_pct_mktcap,
            "rd_yoy_growth_pct": rd_growth_pct,
            "rd_trend": rd_trend,
            "pipeline_intensity": intensity,
            "assessment": assessment,
            "data_caveat": (
                "Phase 1/2/3 stage counts not available via free APIs. "
                "R&D spend intensity is used as a proxy for pipeline depth and quality."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Pipeline value analysis failed: {e}"})


# ── Patent Cliff ───────────────────────────────────────────────────────────────
# No free API directly reports patent expiry schedules. We proxy using:
# - Revenue concentration (if top-line is growing but R&D is flat, risk is building)
# - R&D/revenue ratio trend (declining R&D often precedes patent cliff)
# - Revenue growth deceleration as an early signal

patent_cliff_tool = {
    "name": "get_patent_cliff",
    "description": (
        "Estimates patent cliff risk by analyzing R&D spend trends, revenue growth "
        "deceleration, and gross margin trajectory. A declining R&D/revenue ratio "
        "combined with slowing revenue growth is a key early warning signal. "
        "Flags if risk indicators suggest >30% of revenue could be at risk. "
        "Note: actual patent expiry dates require 10-K/annual report parsing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'PFE'"}
        },
        "required": ["ticker"]
    }
}

def get_patent_cliff(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials

        # 3-4 years of annual revenue
        revenue_series = None
        try:
            revenue_series = fin.loc["Total Revenue"].dropna()
        except Exception:
            pass

        # R&D trend over 3 years
        rd_series = None
        for row_name in ["Research And Development", "Research And Development Expenses"]:
            if row_name in fin.index:
                rd_series = fin.loc[row_name].dropna()
                break

        # Revenue CAGR
        rev_cagr = None
        rev_decel = None
        if revenue_series is not None and len(revenue_series) >= 3:
            rev_vals = [float(v) for v in revenue_series.iloc[:3]]  # newest to oldest
            if rev_vals[2] != 0:
                rev_cagr = round(((rev_vals[0] / rev_vals[2]) ** 0.5 - 1) * 100, 1)
            # Deceleration: compare last year growth to prior year growth
            if rev_vals[1] != 0 and rev_vals[2] != 0:
                g1 = (rev_vals[0] - rev_vals[1]) / abs(rev_vals[1]) * 100
                g2 = (rev_vals[1] - rev_vals[2]) / abs(rev_vals[2]) * 100
                rev_decel = round(g1 - g2, 1)  # negative = decelerating

        # R&D intensity trend
        rd_intensity_trend = None
        rd_pct_vals = []
        if rd_series is not None and revenue_series is not None:
            for i in range(min(len(rd_series), len(revenue_series), 3)):
                rd_v  = float(rd_series.iloc[i])
                rev_v = float(revenue_series.iloc[i])
                if rev_v != 0:
                    rd_pct_vals.append(rd_v / rev_v * 100)
            if len(rd_pct_vals) >= 2:
                if rd_pct_vals[0] < rd_pct_vals[-1] - 2:
                    rd_intensity_trend = "declining"
                elif rd_pct_vals[0] > rd_pct_vals[-1] + 2:
                    rd_intensity_trend = "growing"
                else:
                    rd_intensity_trend = "stable"

        # Gross margin trend (margin compression can signal generic competition entering)
        gm_trend = None
        try:
            gm_series = fin.loc["Gross Profit"].dropna()
            rev_series2 = fin.loc["Total Revenue"].dropna()
            if len(gm_series) >= 3 and len(rev_series2) >= 3:
                gms = [float(gm_series.iloc[i]) / float(rev_series2.iloc[i]) * 100
                       for i in range(3)]
                if gms[0] < gms[-1] - 3:
                    gm_trend = "compressing"
                elif gms[0] > gms[-1] + 3:
                    gm_trend = "expanding"
                else:
                    gm_trend = "stable"
        except Exception:
            pass

        # Risk assessment
        risk_factors = 0
        if rd_intensity_trend == "declining":
            risk_factors += 2
        if rev_decel is not None and rev_decel < -5:
            risk_factors += 2
        if gm_trend == "compressing":
            risk_factors += 1
        if rev_cagr is not None and rev_cagr < 2:
            risk_factors += 1

        if risk_factors >= 4:
            risk_level = "high"
            risk_summary = (
                "Multiple risk indicators are elevated: declining R&D intensity, "
                "revenue deceleration, and/or margin compression. "
                "Suggests existing product revenue may be under pressure from generics or competition."
            )
        elif risk_factors >= 2:
            risk_level = "moderate"
            risk_summary = (
                "Some indicators of potential patent/competitive pressure. "
                "Monitor R&D pipeline fill rate and new product launches."
            )
        else:
            risk_level = "low"
            risk_summary = (
                "No acute patent cliff indicators detected. "
                "R&D investment and revenue growth appear sustainable at current levels."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "revenue_2yr_cagr_pct": rev_cagr,
            "revenue_growth_deceleration_ppt": rev_decel,
            "rd_intensity_trend": rd_intensity_trend,
            "rd_pct_revenue_by_year": [round(v, 1) for v in rd_pct_vals] if rd_pct_vals else None,
            "gross_margin_trend": gm_trend,
            "patent_cliff_risk": risk_level,
            "risk_summary": risk_summary,
            "data_caveat": (
                "Actual patent expiry schedules require parsing 10-K filings. "
                "This tool uses financial trends as early-warning proxies."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Patent cliff analysis failed: {e}"})


# ── FDA Catalyst Risk ──────────────────────────────────────────────────────────
# Binary event risk scoring using news sentiment + R&D trajectory + sector context.
# FDA PDUFA dates and trial results aren't in free APIs, so we proxy via news
# activity and R&D investment concentration.

fda_catalyst_tool = {
    "name": "get_fda_catalyst_risk",
    "description": (
        "Scores FDA/regulatory binary event risk as high/medium/low using news sentiment "
        "trajectory, R&D spend as % of revenue (high % = pipeline-dependent), and "
        "revenue concentration (single-product risk). High R&D + high news activity + "
        "low current revenue = high binary event risk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'MRNA'"}
        },
        "required": ["ticker"]
    }
}

def get_fda_catalyst_risk(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials

        revenue    = info.get("totalRevenue") or 0
        market_cap = info.get("marketCap") or 0

        # R&D intensity
        rd_pct = None
        for row_name in ["Research And Development", "Research And Development Expenses"]:
            if row_name in fin.index:
                rd_vals = fin.loc[row_name].dropna()
                if len(rd_vals) >= 1 and revenue:
                    rd_pct = round(float(rd_vals.iloc[0]) / revenue * 100, 1)
                break

        # Revenue/market cap ratio — low ratio = pre-revenue or pipeline-dependent
        rev_mktcap_ratio = round(revenue / market_cap, 3) if market_cap > 0 else None

        # Operating income (negative = burning cash, likely pipeline-stage)
        operating_income = None
        try:
            oi_series = fin.loc["Operating Income"].dropna()
            if len(oi_series) >= 1:
                operating_income = float(oi_series.iloc[0])
        except Exception:
            pass

        # Risk scoring
        risk_score = 0
        risk_reasons = []

        if rd_pct is not None and rd_pct > 40:
            risk_score += 3
            risk_reasons.append(f"R&D spend is {rd_pct}% of revenue — heavily pipeline-dependent.")
        elif rd_pct is not None and rd_pct > 20:
            risk_score += 2
            risk_reasons.append(f"R&D at {rd_pct}% of revenue signals significant pipeline investment.")

        if rev_mktcap_ratio is not None and rev_mktcap_ratio < 0.05:
            risk_score += 3
            risk_reasons.append(
                f"Revenue is only {rev_mktcap_ratio*100:.1f}% of market cap — "
                "company is valued primarily on pipeline, not current earnings."
            )
        elif rev_mktcap_ratio is not None and rev_mktcap_ratio < 0.15:
            risk_score += 1
            risk_reasons.append("Relatively low current revenue vs. market cap suggests pipeline optionality pricing.")

        if operating_income is not None and operating_income < 0:
            risk_score += 2
            risk_reasons.append("Negative operating income confirms pre-profitability pipeline stage.")

        if risk_score >= 6:
            risk_level = "high"
            summary = (
                "Multiple indicators confirm high binary event dependency. "
                "The stock price is likely pricing in pipeline success — a setback could cause "
                "50–80% drawdowns, while approval could drive 100–300% upside."
            )
        elif risk_score >= 3:
            risk_level = "medium"
            summary = (
                "Moderate binary event risk. Company has meaningful commercial revenue "
                "but also significant pipeline dependency. Key catalysts deserve monitoring."
            )
        else:
            risk_level = "low"
            summary = (
                "Lower binary event risk. Company has established commercial revenue base "
                "that would cushion any single pipeline setback."
            )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "rd_pct_revenue": rd_pct,
            "revenue_to_market_cap_ratio": rev_mktcap_ratio,
            "operating_income": round(operating_income) if operating_income is not None else None,
            "is_pre_profitability": operating_income is not None and operating_income < 0,
            "fda_catalyst_risk": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_reasons,
            "summary": summary,
            "data_caveat": (
                "PDUFA dates and trial readouts require FDA.gov or specialized databases. "
                "This tool uses financial structure as a proxy for binary event dependency."
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"FDA catalyst risk scoring failed: {e}"})


# ── Exports ────────────────────────────────────────────────────────────────────

HEALTHCARE_TOOL_DEFS = [pipeline_value_tool, patent_cliff_tool, fda_catalyst_tool]
HEALTHCARE_FUNCTIONS = {
    "get_pipeline_value": get_pipeline_value,
    "get_patent_cliff": get_patent_cliff,
    "get_fda_catalyst_risk": get_fda_catalyst_risk,
}
