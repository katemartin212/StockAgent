#!/usr/bin/env python3
"""
profile_analysis.py — Standalone profiler for the stock research pipeline.

Runs every tool and data source individually with timing and shows a ranked
breakdown. Run before optimizing to find the real bottlenecks.

Usage:
    .venv/bin/python3 profile_analysis.py NVDA
    .venv/bin/python3 profile_analysis.py NVDA MU
"""

import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
TICKERS = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else ["NVDA"]

results = []

def timed(label, fn, *args, **kwargs):
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        ms = round((time.time() - t0) * 1000)
        error = None
        if isinstance(out, dict) and "error" in out:
            error = out["error"]
        elif isinstance(out, str):
            try:
                d = json.loads(out)
                if "error" in d:
                    error = d["error"]
            except Exception:
                pass
        results.append({"label": label, "ms": ms, "error": error})
        status = f"  ERROR: {error}" if error else ""
        print(f"  {'OK' if not error else 'ERR':3}  {ms:6}ms  {label}{status}")
        return out
    except Exception as e:
        ms = round((time.time() - t0) * 1000)
        results.append({"label": label, "ms": ms, "error": str(e)})
        print(f"  EXC  {ms:6}ms  {label}  EXCEPTION: {e}")
        return None


print(f"\n{'='*60}")
print(f"PROFILING: {TICKER}")
print(f"{'='*60}\n")

# ── Server-side tools (sequential in current code) ────────────────────────────
print("[ STEP 1 ] Sector profile")
from tools_universal import get_sector_profile, get_sector_behavioral_biases
r = timed("get_sector_profile", get_sector_profile, TICKER)
sector = None
if r:
    try:
        sector = json.loads(r).get("sector")
    except Exception:
        pass
print(f"  → sector: {sector}\n")

print("[ STEP 2 ] Sector behavioral biases (uses sector, not ticker)")
timed("get_sector_behavioral_biases", get_sector_behavioral_biases, sector or "_default")
print()

print("[ STEP 3 ] Universal ticker tools (7 — currently SEQUENTIAL)")
from tools_universal import (
    get_stock_price, get_company_info, get_financial_data,
    get_macro_sensitivity, get_insider_activity,
    get_dcf_implied_growth, get_dilution_rate,
)
timed("get_stock_price",        get_stock_price,        TICKER)
timed("get_company_info",       get_company_info,       TICKER)
timed("get_financial_data",     get_financial_data,     TICKER)
timed("get_macro_sensitivity",  get_macro_sensitivity,  TICKER)
timed("get_insider_activity",   get_insider_activity,   TICKER)
timed("get_dcf_implied_growth", get_dcf_implied_growth, TICKER)
timed("get_dilution_rate",      get_dilution_rate,      TICKER)
print()

print("[ STEP 4 ] Sector-specific tools (Technology — 4, currently SEQUENTIAL)")
from tools_tech import (
    get_news_sentiment, get_earnings_surprise,
    get_reddit_sentiment, get_net_revenue_retention,
)
timed("get_news_sentiment",        get_news_sentiment,        TICKER)
timed("get_earnings_surprise",     get_earnings_surprise,     TICKER)
timed("get_reddit_sentiment",      get_reddit_sentiment,      TICKER)
timed("get_net_revenue_retention", get_net_revenue_retention, TICKER)
print()

print("[ STEP 5 ] Parallel: master_signal 8 sources + predictive analytics")
print("  (master_signal already parallel internally — timing the wall-clock)")
from master_signal import get_master_analysis
t_parallel_start = time.time()
master = timed("get_master_analysis (wall clock)", get_master_analysis, TICKER, sector)
if master and "data_freshness_ms" in master:
    print("\n  ── Individual source timings (from master_signal) ──")
    for src, ms_val in sorted(master["data_freshness_ms"].items(), key=lambda x: -x[1]):
        print(f"       {ms_val:6}ms  {src}")

from predictive_analytics import run_all_predictive
timed("run_all_predictive (wall clock)", run_all_predictive, TICKER, sector)
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"{'='*60}")
print("SUMMARY — ranked by duration (slowest first)")
print(f"{'='*60}")
ok_results = [r for r in results if r["error"] is None]
err_results = [r for r in results if r["error"] is not None]
for r in sorted(ok_results, key=lambda x: -x["ms"]):
    bar = "█" * min(40, r["ms"] // 100)
    print(f"  {r['ms']:6}ms  {bar}  {r['label']}")

total_sequential = sum(r["ms"] for r in results if not r["label"].startswith("get_master") and not r["label"].startswith("run_all"))
total_actual     = sum(r["ms"] for r in results)

print(f"\n  Sequential tool time (steps 1-4): {total_sequential:,}ms")
print(f"  Total (all steps, no overlap):    {total_actual:,}ms")

if err_results:
    print(f"\n  ── Errors ──")
    for r in err_results:
        print(f"  ERR  {r['ms']:6}ms  {r['label']}  →  {r['error']}")
