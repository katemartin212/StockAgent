#!/usr/bin/env python3
"""
agent_core.py
-------------
Multi-sector agent loop with automatic sector detection and tool routing.

Usage:
    from agent_core import run_agent
    run_agent("Give me a full analysis of JNJ")

The agent:
  1. First calls get_sector_profile to determine the company's sector
  2. Selects the appropriate sector-specific tool set from SECTOR_TOOL_MAP
  3. Runs the Claude agentic loop with only the relevant tools available
  4. Produces a structured narrative vs. reality report
"""

import json
from datetime import datetime

# ── Import all tools ───────────────────────────────────────────────────────────

from tools_universal import (
    client, _today,
    calculator_tool, calculator,
    sector_profile_tool, get_sector_profile,
    stock_price_tool, get_stock_price,
    company_info_tool, get_company_info,
    financial_data_tool, get_financial_data,
    macro_sensitivity_tool, get_macro_sensitivity,
    insider_activity_tool, get_insider_activity,
    sector_behavioral_biases_tool, get_sector_behavioral_biases,
    dcf_implied_tool, get_dcf_implied_growth,
    dilution_tool, get_dilution_rate,
)
from tools_tech import (
    news_sentiment_tool, get_news_sentiment,
    earnings_surprise_tool, get_earnings_surprise,
    reddit_sentiment_tool, get_reddit_sentiment,
    nrr_tool, get_net_revenue_retention,
)
from tools_healthcare import (
    pipeline_value_tool, get_pipeline_value,
    patent_cliff_tool, get_patent_cliff,
    fda_catalyst_tool, get_fda_catalyst_risk,
)
from tools_financials import (
    nim_tool, get_net_interest_margin,
    llp_tool, get_loan_loss_provisions,
    efficiency_ratio_tool, get_efficiency_ratio,
)
from tools_consumer import (
    sss_tool, get_same_store_sales,
    inventory_turns_tool, get_inventory_turns,
    gm_channel_tool, get_gross_margin_by_channel,
)
from tools_energy import (
    break_even_tool, get_break_even_price,
    reserve_replacement_tool, get_reserve_replacement,
)
from tools_realestate import (
    ffo_tool, get_ffo,
    cap_rate_tool, get_cap_rate,
)
from tools_industrials import (
    book_to_bill_tool, get_book_to_bill,
    capacity_util_tool, get_capacity_utilization,
)

# ── Sector → Tool Map ──────────────────────────────────────────────────────────
# Universal tools (run for EVERY sector, in this order):

_UNIVERSAL = [
    (sector_profile_tool,            get_sector_profile),
    (stock_price_tool,               get_stock_price),
    (company_info_tool,              get_company_info),
    (financial_data_tool,            get_financial_data),
    (macro_sensitivity_tool,         get_macro_sensitivity),
    (insider_activity_tool,          get_insider_activity),
    (sector_behavioral_biases_tool,  get_sector_behavioral_biases),
    (dcf_implied_tool,               get_dcf_implied_growth),
    (dilution_tool,                  get_dilution_rate),
    (calculator_tool,                calculator),
]

# Sector-specific additions (appended after universal):
SECTOR_TOOL_MAP: dict[str, list] = {
    "Technology": [
        (news_sentiment_tool,    get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
        (reddit_sentiment_tool,  get_reddit_sentiment),
        (nrr_tool,               get_net_revenue_retention),
    ],
    "Communication Services": [
        (news_sentiment_tool,    get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
        (reddit_sentiment_tool,  get_reddit_sentiment),
        (nrr_tool,               get_net_revenue_retention),
    ],
    "Healthcare": [
        (pipeline_value_tool,    get_pipeline_value),
        (patent_cliff_tool,      get_patent_cliff),
        (fda_catalyst_tool,      get_fda_catalyst_risk),
        (news_sentiment_tool,    get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
    ],
    "Financial Services": [
        (nim_tool,               get_net_interest_margin),
        (llp_tool,               get_loan_loss_provisions),
        (efficiency_ratio_tool,  get_efficiency_ratio),
        (news_sentiment_tool,    get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
    ],
    "Consumer Cyclical": [
        (sss_tool,               get_same_store_sales),
        (inventory_turns_tool,   get_inventory_turns),
        (gm_channel_tool,        get_gross_margin_by_channel),
        (news_sentiment_tool,    get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
    ],
    "Consumer Defensive": [
        (sss_tool,               get_same_store_sales),
        (inventory_turns_tool,   get_inventory_turns),
        (gm_channel_tool,        get_gross_margin_by_channel),
        (news_sentiment_tool,    get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
    ],
    "Energy": [
        (break_even_tool,            get_break_even_price),
        (reserve_replacement_tool,   get_reserve_replacement),
        (news_sentiment_tool,        get_news_sentiment),
        (earnings_surprise_tool,     get_earnings_surprise),
    ],
    "Real Estate": [
        (ffo_tool,               get_ffo),
        (cap_rate_tool,          get_cap_rate),
        (news_sentiment_tool,    get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
    ],
    "Industrials": [
        (book_to_bill_tool,     get_book_to_bill),
        (capacity_util_tool,    get_capacity_utilization),
        (news_sentiment_tool,   get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
    ],
    # Fallback for Basic Materials, Utilities, unknown sectors
    "_default": [
        (news_sentiment_tool,    get_news_sentiment),
        (earnings_surprise_tool, get_earnings_surprise),
    ],
}


def _build_tool_set(sector: str | None) -> tuple[list, dict]:
    """
    Combine universal tools with sector-specific tools.
    Returns (tool_defs_list, functions_dict).
    """
    sector_key  = sector if sector in SECTOR_TOOL_MAP else "_default"
    combined    = _UNIVERSAL + SECTOR_TOOL_MAP[sector_key]
    tool_defs   = [t for t, _ in combined]
    functions   = {t["name"]: fn for t, fn in combined}
    return tool_defs, functions


def _detect_sector(ticker: str) -> str | None:
    """Quick synchronous sector lookup before starting the agent loop."""
    try:
        import yfinance as yf
        return yf.Ticker(ticker.upper()).info.get("sector")
    except Exception:
        return None


# ── System Prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt(sector: str | None) -> str:
    sector_label = sector or "Unknown"
    tool_count   = len(_UNIVERSAL) + len(SECTOR_TOOL_MAP.get(sector or "_default", SECTOR_TOOL_MAP["_default"]))

    sector_guidance = {
        "Technology": (
            "ALWAYS run get_dcf_implied_growth and get_dilution_rate — these are the most important "
            "valuation tools for IT stocks where SBC and growth expectations are central to the debate. "
            "Run get_net_revenue_retention for any SaaS/cloud/subscription company."
        ),
        "Healthcare": (
            "ALWAYS run get_pipeline_value, get_patent_cliff, and get_fda_catalyst_risk. "
            "The narrative vs reality analysis must address binary FDA event risk explicitly — "
            "retail investors chronically overprice approval probability."
        ),
        "Financial Services": (
            "ALWAYS run get_net_interest_margin, get_loan_loss_provisions, and get_efficiency_ratio. "
            "NIM trend and provision trajectory are the two most important leading indicators for banks. "
            "Never use P/E as the primary valuation metric for financial stocks — use P/Book or P/TBV."
        ),
        "Consumer Cyclical": (
            "ALWAYS run get_same_store_sales, get_inventory_turns, and get_gross_margin_by_channel. "
            "Inventory turns are a leading indicator — declining turns signal demand weakness before "
            "it shows in revenue. Comps vs total revenue growth is the key retail metric."
        ),
        "Consumer Defensive": (
            "Run get_same_store_sales and get_inventory_turns. Focus on volume vs pricing mix — "
            "pricing-driven revenue growth is weaker and more fragile than volume growth."
        ),
        "Energy": (
            "ALWAYS run get_break_even_price and get_reserve_replacement. "
            "The analysis must address commodity price sensitivity and reserve life explicitly. "
            "Never evaluate energy stocks at peak commodity prices — stress-test at mid-cycle."
        ),
        "Real Estate": (
            "ALWAYS run get_ffo and get_cap_rate. "
            "NEVER use P/E or GAAP earnings to value REITs — P/FFO is the correct multiple. "
            "Cap rate vs. 10-year Treasury spread is the key valuation anchor for REITs."
        ),
        "Industrials": (
            "ALWAYS run get_book_to_bill and get_capacity_utilization. "
            "Revenue momentum and operating leverage are the primary signals — "
            "book-to-bill > 1.0 is a strong leading indicator of future revenue acceleration."
        ),
    }.get(sector_label, (
        "Use get_dcf_implied_growth and get_dilution_rate as primary valuation context tools. "
        "Supplement with news sentiment and earnings history."
    ))

    return f"""You are a senior equity analyst at a multi-sector quantitative investment fund.
Today's date is {_today}. You are analyzing a {sector_label} sector company.
You have access to {tool_count} tools: 9 universal tools that run for every sector,
plus sector-specific tools calibrated for {sector_label}.

SECTOR GUIDANCE — {sector_label}:
{sector_guidance}

GENERAL RULES:
1. Call get_sector_profile first if you haven't already identified the sector.
2. Call get_sector_behavioral_biases with the detected sector to frame the narrative analysis.
3. Never guess at data — call the relevant tool and use the exact numbers returned.
4. Use the calculator for any derived metric.
5. Always produce output in the structured report format below.

──────────────────────────────────────────────────────────────
REPORT FORMAT
──────────────────────────────────────────────────────────────

# {{TICKER}} — {{Company Name}}
*{_today} | {{Sector}} · {{Industry}}*
*Analysis toolkit: {{list the sector-specific tools used}}*

---

## 1. Fundamentals
Core financial metrics from get_financial_data + sector-specific metrics.

## 2. Sector-Specific Analysis
Results from the {sector_label} toolkit — the metrics that matter most for this sector.

## 3. Valuation Intelligence
DCF-implied growth rate vs. historical delivery. SBC/dilution impact on true FCF.

## 4. Macro Sensitivity
From get_macro_sensitivity: rate, USD, GDP sensitivity ratings and explanations.

## 5. Insider Activity (Last 90 Days)
From get_insider_activity: Form 4 filing count and activity signal.

## 6. News Sentiment & Earnings Track Record
(If applicable to the sector toolkit)

---

## 🎯 Narrative vs Reality

**🗣️ Retail Narrative:** What retail investors appear to believe (informed by get_sector_behavioral_biases).
Include 3–5 specific claims drawn from the sector biases, each fact-checked against live data.

**📊 Fundamental Reality:** What the hard data actually shows.

**⚡ Divergence Score: X / 100**
> 0–20 Aligned | 21–40 Mild gap | 41–60 Notable | 61–80 Strong | 81–100 Extreme

**🏷️ Recommended Action:** ACCUMULATE / HOLD / TRIM / AVOID

**💡 Rationale:** 2–3 sentences grounded in the divergence between sector-specific metrics and retail narrative.

---
*Sources: Yahoo Finance (live) · SEC EDGAR Form 4 API (free) · Reddit public API*
"""


# ── Agent Loop ─────────────────────────────────────────────────────────────────

def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    Run the multi-sector stock research agent on a query.
    Automatically detects sector and routes to the appropriate tool set.
    Loops until Claude stops requesting tool calls. Max 25 steps.
    """
    # Extract ticker hint from message for sector pre-detection
    # (Claude will call get_sector_profile properly during the loop)
    import re
    ticker_match = re.search(r'\b([A-Z]{1,5})\b', user_message.upper())
    sector_hint  = _detect_sector(ticker_match.group(1)) if ticker_match else None

    tool_defs, functions = _build_tool_set(sector_hint)
    system_prompt        = _build_system_prompt(sector_hint)

    sector_display = sector_hint or "unknown (will auto-detect)"

    if verbose:
        print(f"\n{'═' * 68}")
        print(f"  {user_message}")
        print(f"{'═' * 68}")
        print(f"  Detected sector: {sector_display}")
        print(f"  Tool set: {len(tool_defs)} tools loaded")
        print()

    messages = [{"role": "user", "content": user_message}]
    step = 0

    while True:
        step += 1

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=tool_defs,
            messages=messages,
        )

        if verbose:
            print(f"── Step {step}  stop_reason={response.stop_reason}")

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "text" and block.text.strip() and verbose:
                    snippet = block.text[:200] + "…" if len(block.text) > 200 else block.text
                    print(f"   🧠 {snippet}")

                elif block.type == "tool_use":
                    if verbose:
                        print(f"   🛠  {block.name}({json.dumps(block.input)})")

                    fn     = functions.get(block.name)
                    result = fn(**block.input) if fn else json.dumps({"error": f"Unknown tool: {block.name}"})

                    # Dynamically update tool set if sector was just detected
                    if block.name == "get_sector_profile":
                        try:
                            profile = json.loads(result)
                            new_sector = profile.get("sector")
                            if new_sector and new_sector != sector_hint:
                                tool_defs, functions = _build_tool_set(new_sector)
                                system_prompt = _build_system_prompt(new_sector)
                                if verbose:
                                    print(f"      ↳ Sector detected: {new_sector} — tool set updated")
                        except Exception:
                            pass

                    if verbose:
                        preview = result[:140] + "…" if len(result) > 140 else result
                        print(f"      → {preview}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            answer = "".join(b.text for b in response.content if hasattr(b, "text"))
            print(f"\n{answer}")
            print(f"\n✅ Completed in {step} step(s)")
            return answer

        if step >= 25:
            err = "Error: agent exceeded 25 steps — possible loop. Aborting."
            print(err)
            return err


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — 5 cross-sector test cases
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 68)
    print("  MULTI-SECTOR RESEARCH AGENT — Universal Coverage")
    print("█" * 68)

    tests = [
        (
            "JNJ",
            "Give me a full analysis of Johnson & Johnson (JNJ). "
            "Focus on pipeline optionality, patent cliff risk, and whether the valuation "
            "reflects the FDA catalyst risks in the portfolio.",
        ),
        (
            "JPM",
            "Analyze JPMorgan Chase (JPM). What does the net interest margin trend, "
            "loan loss provision trajectory, and efficiency ratio tell us about "
            "where we are in the credit cycle? Is the valuation attractive?",
        ),
        (
            "HD",
            "What's the fundamental picture for Home Depot (HD)? "
            "Analyze inventory turns, same-store sales trends, and margin trajectory. "
            "Is the current valuation pricing in a housing cycle recovery or contraction?",
        ),
        (
            "SPG",
            "Is Simon Property Group (SPG) a good REIT investment right now? "
            "Analyze FFO, cap rate, and what the implied yield says about valuation "
            "relative to the current interest rate environment.",
        ),
        (
            "NVDA vs XOM",
            "Compare NVIDIA (NVDA) and ExxonMobil (XOM) — which is the better investment "
            "right now? Analyze each using their sector-appropriate toolkit, then synthesize "
            "a cross-sector comparison based on valuation, fundamental quality, "
            "and risk-adjusted return potential.",
        ),
    ]

    for ticker_hint, question in tests:
        print(f"\n\n{'▓' * 68}")
        print(f"  TEST: {ticker_hint}")
        print("▓" * 68)
        run_agent(question)
