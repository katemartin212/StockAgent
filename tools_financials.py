#!/usr/bin/env python3
"""
tools_financials.py
-------------------
Financial Services sector tools (banks, insurance, fintech):
  - get_net_interest_margin  — NII / average earning assets; core bank profitability
  - get_loan_loss_provisions — provision for credit losses as % of loans; early warning
  - get_efficiency_ratio     — non-interest expense / revenue; operating leverage
"""

import json
import yfinance as yf

# ── Net Interest Margin ────────────────────────────────────────────────────────
# NIM = Net Interest Income / Average Earning Assets
# yfinance exposes Net Interest Income for banks from the income statement.
# Total Assets from the balance sheet serves as a proxy for earning assets.

nim_tool = {
    "name": "get_net_interest_margin",
    "description": (
        "Net Interest Margin (NIM) = Net Interest Income / Average Earning Assets. "
        "This is the core profitability metric for banks and financial institutions. "
        "Returns NIM trend over 4 quarters, YoY change, and benchmark context. "
        "A NIM above 3.5% is strong for US banks; below 2.5% signals compression risk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'JPM'"}
        },
        "required": ["ticker"]
    }
}

def get_net_interest_margin(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials

        # Net Interest Income — annual
        nii_annual = None
        for row_name in ["Net Interest Income", "Net Interest Income After Provision For Loan Losses"]:
            if row_name in fin.index:
                nii_series = fin.loc[row_name].dropna()
                if len(nii_series) >= 1:
                    nii_annual = float(nii_series.iloc[0])
                break

        if nii_annual is None:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": (
                    "Net Interest Income not found. This company may not be a bank or "
                    "traditional financial institution where NIM is applicable."
                ),
                "data_source": "Yahoo Finance via yfinance",
            })

        # Total assets as proxy for earning assets
        total_assets = None
        try:
            bs = stock.balance_sheet
            for row_name in ["Total Assets", "Total Assets Net Minority Interest"]:
                if row_name in bs.index:
                    ta_series = bs.loc[row_name].dropna()
                    if len(ta_series) >= 1:
                        total_assets = float(ta_series.iloc[0])
                    break
        except Exception:
            pass

        nim_pct = round(nii_annual / total_assets * 100, 2) if total_assets else None

        # YoY NIM change
        nim_prior = None
        try:
            nii_series2 = fin.loc["Net Interest Income"].dropna()
            bs2 = stock.balance_sheet
            ta_series2 = bs2.loc["Total Assets"].dropna() if "Total Assets" in bs2.index else None
            if len(nii_series2) >= 2 and ta_series2 is not None and len(ta_series2) >= 2:
                nim_prior = round(float(nii_series2.iloc[1]) / float(ta_series2.iloc[1]) * 100, 2)
        except Exception:
            pass

        nim_change = round(nim_pct - nim_prior, 2) if (nim_pct and nim_prior) else None

        # Benchmark
        if nim_pct is None:
            benchmark = "NIM could not be calculated — total assets data unavailable."
        elif nim_pct > 3.5:
            benchmark = f"Strong NIM of {nim_pct}% — above the 3.5% threshold for well-positioned banks."
        elif nim_pct > 2.5:
            benchmark = f"Adequate NIM of {nim_pct}% — in-line with typical US commercial bank range."
        else:
            benchmark = f"NIM of {nim_pct}% is below 2.5% — signals margin compression or low-yield asset mix."

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "net_interest_income_annual": round(nii_annual),
            "total_assets": round(total_assets) if total_assets else None,
            "nim_pct": nim_pct,
            "nim_prior_year_pct": nim_prior,
            "nim_yoy_change_ppt": nim_change,
            "nim_trend": (
                "expanding" if nim_change and nim_change > 0.1
                else "compressing" if nim_change and nim_change < -0.1
                else "stable"
            ) if nim_change is not None else "unknown",
            "benchmark": benchmark,
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Net interest margin calculation failed: {e}"})


# ── Loan Loss Provisions ───────────────────────────────────────────────────────
# Provision for Credit Losses (PCL) as % of total loans.
# Rising provisions are the earliest warning signal for credit cycle deterioration —
# they appear 2–4 quarters before charge-offs hit.

llp_tool = {
    "name": "get_loan_loss_provisions",
    "description": (
        "Provision for Credit Losses as % of Total Loans, trended over 4 years. "
        "Rising provisions are an early warning of credit deterioration that appears "
        "2–4 quarters before actual loan losses show up. Above 1% is elevated; "
        "above 2% signals significant stress. Below 0.3% in a normal cycle is lean."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'BAC'"}
        },
        "required": ["ticker"]
    }
}

def get_loan_loss_provisions(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials
        bs     = stock.balance_sheet

        # Provision for credit losses
        provision = None
        for row_name in ["Credit Losses Provision", "Provision For Loan Losses",
                         "Provision For Doubtful Accounts", "Provision And Write Offs"]:
            if row_name in fin.index:
                prov_series = fin.loc[row_name].dropna()
                if len(prov_series) >= 1:
                    provision = float(prov_series.iloc[0])
                break

        if provision is None:
            # Try to find via net income and pre-provision income
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "Provision for Credit Losses not found. May not be a lending institution.",
                "data_source": "Yahoo Finance via yfinance",
            })

        # Total loans / receivables from balance sheet
        total_loans = None
        for row_name in ["Net Loan", "Net Receivables", "Loans Net",
                         "Gross Accounts Receivable", "Receivables"]:
            if row_name in bs.index:
                loan_series = bs.loc[row_name].dropna()
                if len(loan_series) >= 1:
                    total_loans = float(loan_series.iloc[0])
                break

        provision_pct = round(provision / total_loans * 100, 3) if total_loans else None

        # Multi-year trend
        provision_trend = []
        try:
            prov_s = fin.loc["Credit Losses Provision"].dropna() if "Credit Losses Provision" in fin.index else None
            loans_s = bs.loc["Net Loan"].dropna() if "Net Loan" in bs.index else None
            if prov_s is not None and loans_s is not None:
                for i in range(min(len(prov_s), len(loans_s), 4)):
                    p = float(prov_s.iloc[i])
                    l = float(loans_s.iloc[i])
                    if l != 0:
                        provision_trend.append(round(p / l * 100, 3))
        except Exception:
            pass

        # Trend direction
        trend_dir = "unknown"
        if len(provision_trend) >= 2:
            if provision_trend[0] > provision_trend[-1] * 1.2:
                trend_dir = "rising"
            elif provision_trend[0] < provision_trend[-1] * 0.8:
                trend_dir = "falling"
            else:
                trend_dir = "stable"

        if provision_pct is None:
            benchmark = "Cannot calculate provision rate — loan data unavailable."
        elif provision_pct > 2.0:
            benchmark = f"Provision rate of {provision_pct}% is very high — signals significant credit stress."
        elif provision_pct > 1.0:
            benchmark = f"Provision rate of {provision_pct}% is elevated — above normal cycle levels."
        elif provision_pct > 0.3:
            benchmark = f"Provision rate of {provision_pct}% is within normal cycle range (0.3–1.0%)."
        else:
            benchmark = f"Provision rate of {provision_pct}% is lean — could indicate benign credit or under-provisioning."

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "provision_for_credit_losses": round(provision),
            "total_loans": round(total_loans) if total_loans else None,
            "provision_pct_of_loans": provision_pct,
            "provision_trend_pct": provision_trend,
            "trend_direction": trend_dir,
            "benchmark": benchmark,
            "signal": (
                "warning" if trend_dir == "rising" and (provision_pct or 0) > 0.5
                else "elevated" if (provision_pct or 0) > 1.0
                else "normal"
            ),
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Loan loss provisions analysis failed: {e}"})


# ── Efficiency Ratio ───────────────────────────────────────────────────────────
# Efficiency Ratio = Non-Interest Expense / Net Revenue
# For banks: Net Revenue = Net Interest Income + Non-Interest Income
# Below 60% is good; below 50% is excellent; above 70% signals structural cost issues.

efficiency_ratio_tool = {
    "name": "get_efficiency_ratio",
    "description": (
        "Bank Efficiency Ratio = Non-Interest Expense / Net Revenue. "
        "Below 60% is considered good for US banks; below 50% is excellent; "
        "above 70% indicates structural cost problems. Returns efficiency ratio, "
        "trend over 3 years, and peer benchmark assessment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'WFC'"}
        },
        "required": ["ticker"]
    }
}

def get_efficiency_ratio(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        fin    = stock.financials

        # Non-interest expense (operating expense for banks)
        op_expense = None
        for row_name in ["Non Interest Expense", "Operating Expense",
                         "Total Operating Expenses", "Non-Interest Expense"]:
            if row_name in fin.index:
                exp_series = fin.loc[row_name].dropna()
                if len(exp_series) >= 1:
                    op_expense = abs(float(exp_series.iloc[0]))  # absolute value
                break

        # Net revenue for banks = NII + non-interest income
        # Proxy: use Total Revenue or Operating Revenue
        net_revenue = None
        for row_name in ["Total Revenue", "Net Interest Income", "Operating Revenue"]:
            if row_name in fin.index:
                rev_series = fin.loc[row_name].dropna()
                if len(rev_series) >= 1:
                    net_revenue = float(rev_series.iloc[0])
                break

        if not op_expense or not net_revenue:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": "Operating expense or revenue data unavailable for efficiency ratio calculation.",
                "data_source": "Yahoo Finance via yfinance",
            })

        efficiency = round(op_expense / abs(net_revenue) * 100, 1)

        # Multi-year trend
        efficiency_trend = []
        try:
            exp_s = fin.loc["Non Interest Expense"].dropna() if "Non Interest Expense" in fin.index else (
                fin.loc["Operating Expense"].dropna() if "Operating Expense" in fin.index else None
            )
            rev_s = fin.loc["Total Revenue"].dropna() if "Total Revenue" in fin.index else None
            if exp_s is not None and rev_s is not None:
                for i in range(min(len(exp_s), len(rev_s), 3)):
                    e = abs(float(exp_s.iloc[i]))
                    r = abs(float(rev_s.iloc[i]))
                    if r != 0:
                        efficiency_trend.append(round(e / r * 100, 1))
        except Exception:
            pass

        trend_dir = "unknown"
        if len(efficiency_trend) >= 2:
            if efficiency_trend[0] > efficiency_trend[-1] + 2:
                trend_dir = "deteriorating"
            elif efficiency_trend[0] < efficiency_trend[-1] - 2:
                trend_dir = "improving"
            else:
                trend_dir = "stable"

        if efficiency < 50:
            benchmark = f"Excellent efficiency ratio of {efficiency}% — best-in-class operating leverage."
        elif efficiency < 60:
            benchmark = f"Good efficiency ratio of {efficiency}% — above average for US banks."
        elif efficiency < 70:
            benchmark = f"Adequate efficiency ratio of {efficiency}% — in-line with peer median."
        else:
            benchmark = f"Elevated efficiency ratio of {efficiency}% — signals cost structure needs improvement."

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "non_interest_expense": round(op_expense),
            "net_revenue": round(net_revenue),
            "efficiency_ratio_pct": efficiency,
            "efficiency_trend_pct": efficiency_trend,
            "trend_direction": trend_dir,
            "benchmark": benchmark,
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"Efficiency ratio calculation failed: {e}"})


# ── Exports ────────────────────────────────────────────────────────────────────

FINANCIALS_TOOL_DEFS = [nim_tool, llp_tool, efficiency_ratio_tool]
FINANCIALS_FUNCTIONS = {
    "get_net_interest_margin": get_net_interest_margin,
    "get_loan_loss_provisions": get_loan_loss_provisions,
    "get_efficiency_ratio": get_efficiency_ratio,
}
