# Stock Research Agent — Project Context

Last updated: 2026-04-03

---

## What This Is

A production-grade, multi-sector AI stock research terminal. It runs locally via a FastAPI backend (`server.py`) and a single-file React dashboard (`dashboard.html`). The AI layer uses the Anthropic Claude API (`claude-sonnet-4-6`) to synthesize live data from 8+ free data sources into an institutional-grade research report.

**Python:** 3.13 &nbsp;|&nbsp; **Key packages:** anthropic 0.86.0, yfinance 1.2.0, pandas 3.0.1, scikit-learn 1.8.0, scipy 1.17.1

**To run:**
```bash
.venv/bin/uvicorn server:app --reload
# then open dashboard.html in browser
```

**If port is blocked:**
```bash
lsof -ti:8000 | xargs kill -9
```

---

## File Map

### Core Application
| File | Purpose |
|------|---------|
| `server.py` | FastAPI backend. SSE streaming, `/analyze`, `/comps`, `/validate` endpoints |
| `dashboard.html` | Single-file React + Chart.js frontend (~3200 lines) |
| `master_signal.py` | Aggregates all data sources into a master composite signal |
| `stock_research_agent.py` | Main agent entry point (8-tool version, latest) |
| `stock_agent.py` | Earlier 4-tool version (preserved) |
| `SCENARIO_MODEL_LIMITATIONS.md` | 12 documented limitations of the DCF scenario model with bias direction and severity |

### Predictive Analytics (Predict Tab)
| File | Purpose |
|------|---------|
| `predictive_analytics.py` | 4 forward-looking models: factor attribution, earnings probability, scenario analysis, sentiment mean reversion |
| `predictive/__init__.py` | Package root for predictive low-level helpers |
| `predictive/_timeseries.py` | `align_weekly` (fetch + TZ normalize + resample), `robust_sigma` (MAD-based annualized σ), `zscore_log`, `decay_weights` |
| `predictive/_ridge.py` | `weighted_ridge` — GCV-tuned Ridge regression with sandwich SEs and 95% CIs |
| `validation.py` | Walk-forward out-of-sample validation framework. Produces HIGH/MEDIUM/LOW/UNVALIDATED tier |

### Shared Utilities
| File | Purpose |
|------|---------|
| `tools_base.py` | Shared helpers for all `tools_*.py` modules: `fetch_info`, `safe_ratio`, `_find_series`, `tool_result`, `_tool_schema` (generates standard single-ticker tool definition dicts) |

### Tool Modules
| File | Tools |
|------|-------|
| `tools_universal.py` | Stock price, company info, financials, macro sensitivity, insider activity, DCF implied growth, dilution rate, sector profile, behavioral biases |
| `tools_tech.py` | News sentiment, earnings surprise, Reddit sentiment, NRR |
| `tools_healthcare.py` | Pipeline value, patent cliff, FDA catalyst risk |
| `tools_financials.py` | Net interest margin, loan loss provisions, efficiency ratio |
| `tools_consumer.py` | Same-store sales, inventory turns, gross margin by channel |
| `tools_energy.py` | Break-even price, reserve replacement |
| `tools_realestate.py` | FFO, cap rate |
| `tools_industrials.py` | Book-to-bill, capacity utilization |

### Data Sources (`data_sources/`)
| File | Source |
|------|--------|
| `_cache.py` | Two-layer cache: in-memory + SQLite persistence. Survives server restarts. |
| `_http.py` | Shared HTTP helpers: `get(url)` (requests wrapper with default headers), `get_cached_http(key, url)` (fetch + cache in one call) |
| `sec_edgar.py` | SEC EDGAR free API — financial filings. Exposes `get_edgar_form4` (alias: `get_insider_activity`) |
| `reddit_sentiment.py` | Reddit API (PRAW) — retail sentiment with relevance filtering |
| `stocktwits_sentiment.py` | StockTwits API — trader sentiment |
| `trends_signal.py` | Google Trends via pytrends |
| `fred_macro.py` | FRED St. Louis Fed — macro indicators. Exposes `get_fred_macro` (alias: `get_macro_context`) |
| `comps_data.py` | yfinance peer comparison table |

Note: `open_insider.py` was deleted (2026-04-03) — OpenInsider consistently timed out at the 8s limit. Insider signals now come exclusively from SEC EDGAR Form 4 via `get_insider_activity()`.

---

## Dashboard Tabs

### Snapshot Tab
- Company overview panel (description, products, competitors, next earnings)
- Macro strip (fed funds, 10Y yield, CPI, yield curve)
- Score cards: Fundamental / Retail Sentiment / Divergence / Composite
- Behavioral signal + verdict (ACCUMULATE/HOLD/TRIM/AVOID)
- Retail narrative feed (Reddit + StockTwits posts with fact-checks)
- Fundamentals table (Rule of 40, EV/Rev, FCF margin, NRR, etc.)
- Sector-specific metrics panel
- Comps table (peer valuation comparison, Claude-written verdict)
- Insider activity + reasoning trace

### Predict Tab
- **Model Confidence Panel** — traffic-light validation tier badge; click "run validation" to trigger walk-forward tests
- Factor Attribution — Ridge regression bar chart with FDR-corrected significance, 95% CIs, current headwinds/tailwinds (8-factor model)
- Earnings Surprise Probability — semicircle gauge with bootstrap 95% CI, sub-scores, 8Q history chart
- Scenario Analysis — DCF bear/base/bull cards with FCF year 1–3 table, probability sliders seeded from model outputs
- Sentiment Mean Reversion — z-score line chart with ±1.5σ bands, correlation validation

---

## Predictive Models & Validation

### `predictive_analytics.py` — 4 models

**1. Factor Attribution** (`get_factor_attribution`)
- 8-factor weighted Ridge regression: 10Y yield, DXY, VIX, sector ETF, S&P 500, 4W momentum, 1W reversal, Value/Growth spread (IWD/IWF)
- Weekly returns, trailing 2 years
- Ridge alpha cross-validated via GCV (sklearn RidgeCV, alphas=[0.01, 0.1, 1, 10, 100])
- Sandwich covariance for SEs: σ² · (X'WX + αI)⁻¹ · X'WX · (X'WX + αI)⁻¹
- Benjamini-Hochberg FDR correction on all p-values
- 95% CI on each coefficient
- Current conditions vs. prior-history percentile (look-ahead bias fixed: uses `iloc[:-1]`)

**2. Earnings Surprise Probability** (`get_earnings_surprise_probability`)
- 4 sub-scores: historical beat rate (35%), revision momentum (35%), analyst sentiment (15%), sector read-through (15%)
- Filters to past-only events (look-ahead bias fixed)
- Bootstrap 95% CI on final probability (1000 resamples, seed=42)

**3. Scenario Analysis** (`get_scenario_analysis`) — rebuilt 2026-04-03
- **Architecture:** `_compute_dcf_core()` (cached 4h) → `_apply_behavioral()` (uncached, fresh each call) → `get_scenario_analysis()` wrapper
- **Revenue model:** yfinance `revenue_estimate` annual rows parsed (annual vs. quarterly row disambiguation); falls back to 2-year CAGR from quarterly income statement. Near-term (yrs 1–2) uses consensus ×0.92/×1.0/×1.08 for bear/base/bull. Mid-term (yrs 3–5) fades from consensus-implied Y1→Y2 growth (capped 35%). Terminal (yrs 6–10) linearly decays to 3%.
- **Margin model:** gross margin evolution, operating leverage (opex scales at 95%/100%/80% of revenue growth for bear/base/bull), capex %, SBC %
- **FCF:** Revenue × (gross_margin − opex_margin − capex_pct − sbc_pct) per year, 10-year path
- **Discount rate:** risk-free (10Y Treasury 4.5%) + ERP (4.5% base + (beta−1)×1.5%, floored 4.5%, capped 8.0%)
- **Terminal multiple:** 18× FCF if gross margin > 60%; 14× if 40–60%; 10× if <40%; ±20% for bull/bear
- **Comps cross-check:** implied EV/Rev and EV/EBITDA in each scenario vs. peer median; flags if stretched
- **Behavioral adjustments** (`_apply_behavioral`): narrative adjustment ±10% on base case from divergence_score; earnings surprise ±3% on base revenue; probability model adjusts bear/base/bull from 25/50/25 baseline using divergence, macro, insider, earnings surprise, and sentiment z-score signals
- **Guards:** `is_financial` (banks/insurers → DCF not applicable, returns None prices + flag); `deeply_negative_fcf` (FCF margin yr1 < −50% → returns None prices + flag)
- **bull_mid ordering:** floored at `base_mid × 1.20` to guarantee bear < base < bull when yfinance returns fewer than 8 quarters of quarterly data (e.g., recently-listed tickers)
- **Output:** scenarios dict with year 1–3 projections, FCF assumptions, implied multiples, probability drivers, narrative text, active limitations list; `model_inputs` row (discount rate, terminal multiple, peer median EV/Rev)

**Behavioral inputs convention:** `master_signal.divergence_score` (HIGH = undervalued) is flipped to `100 − divergence_score` before passing to `_apply_behavioral()` so HIGH = overhyped, as the spec requires.

**4. Sentiment Mean Reversion** (`get_sentiment_mean_reversion`)
- 2Y Google Trends weekly z-score vs. 26W baseline
- Baseline uses `iloc[:-2]` to exclude incomplete recent weeks (look-ahead bias fixed)
- Correlation with 4W forward returns

### `run_all_predictive()` — two-phase execution
1. **Phase 1** (parallel): factor attribution, earnings probability, sentiment mean reversion
2. **Phase 2**: scenario analysis, with enriched `behavioral_inputs` from Phase 1 results (earnings surprise probability, sentiment z-score) merged with master_signal outputs from server.py

### `validation.py` — 4 validation tests

**Factor Model validation** (`validate_factor_model`)
- Lagged walk-forward OLS (X[t] predicts y[t+1])
- 5Y weekly data, 52W minimum training window
- Validated if: directional accuracy > 52% AND |IC| > 0.04

**Earnings Model validation** (`validate_earnings_model`)
- Leave-one-out CV on `earnings_dates` (up to 24+ historical events)
- Brier Score and Brier Skill Score vs. 50% naive baseline
- Validated if: brier_skill > 0 AND n_events ≥ 6

**Sentiment Signal validation** (`validate_sentiment_signal`)
- t-test on forward returns conditioned on |z| > 1.5 signals
- Bootstrap 95% CI on mean forward return (1000 resamples)
- Validated if: |t-stat| > 1.96 AND n_signals ≥ 8

**Scenario Model validation** (`validate_scenario_model`) — added 2026-04-03
- Walk-forward backtest over last 3 years at quarterly checkpoints
- At each checkpoint: reconstructs TTM revenue, computes simplified bear/base/bull prices using historical EV/Rev multiple
- Compares probability-weighted target (25% bear + 50% base + 25% bull) to actual price 52 weeks later
- Metrics: MAE%, coverage_rate (fraction of periods where actual price fell within bear–bull range; target ≥ 65%), n_periods
- Validated if: coverage_rate ≥ 0.65 AND n_periods ≥ 6

**Confidence Tiers:**
- HIGH = 3/3 validated (factor + earnings + sentiment; scenario validation runs separately)
- MEDIUM = 2/3 validated
- LOW = 1/3 validated
- UNVALIDATED = 0/3 validated

### Test Results (2026-03-27)
| Ticker | Factor | Earnings | Sentiment | Tier |
|--------|--------|----------|-----------|------|
| NVDA | ✗ (IC 0.02, n.s.) | ✓ (skill 0.71, n=24, 95.8% beat rate) | blocked (Google 400) | LOW |
| QBTS | ✗ (IC -0.02, n.s.) | ✗ (skill -0.15, n=15, 46.7% beat) | blocked | UNVALIDATED |
| MU | ✗ (IC 0.02, n.s.) | ✓ (skill 0.59, n=24, 91.7% beat) | blocked | LOW |

Note: Google Trends returns 400 during test sessions (rate limiting). Sentiment model validates correctly when Trends is accessible.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/analyze` | POST `{ticker}` | SSE stream: runs all tools + Claude synthesis |
| `/comps` | POST `{subject_ticker, peers, sector}` | Peer comparison table + Claude verdict |
| `/validate` | POST `{ticker, sector}` | Walk-forward validation (60-180s, cached 6h) |
| `/prefetch` | POST `{ticker, sources}` | Background-fire slow fetches (edgar/fred/profile). Returns 202 immediately. |
| `/cache/stats` | GET | Hit rate, time saved, key counts (mem + SQLite) |

### SSE Event Types (`/analyze`)
`tool_start` → `tool_done` → `sector_detected` → `parallel_start` → `source_done` → `parallel_done` → `predictive_done` → `synthesizing` → `result` / `error`

---

## Sector Coverage

| Sector | Extra Tools |
|--------|-------------|
| Technology / Comm Services | News sentiment, earnings surprise, Reddit, NRR |
| Healthcare | Pipeline value, patent cliff, FDA catalyst risk, news, earnings |
| Financial Services | NIM, loan loss provisions, efficiency ratio, news, earnings |
| Consumer Cyclical/Defensive | Same-store sales, inventory turns, gross margin by channel, news, earnings |
| Energy | Break-even price, reserve replacement, news, earnings |
| Real Estate | FFO, cap rate, news, earnings |
| Industrials | Book-to-bill, capacity utilization, news, earnings |
| Default (all others) | News + earnings |

---

## Known Limitations / Watch Out For

- **DCF scenario model — financial companies:** Banks and insurers have `grossMargins=0` in yfinance. DCF produces meaningless results; `is_financial` flag returns None prices with an explanatory flag. Comps cross-check should use P/B or P/E for these tickers.
- **DCF scenario model — pre-profitability companies:** If base FCF margin in year 1 < −50% (e.g., QBTS), `deeply_negative_fcf` flag returns None prices. Consider EV/Revenue or milestone-based valuation instead.
- **DCF scenario model — yfinance revenue estimates:** `revenue_estimate` returns rows indexed "0q"/"+1q" (quarterly) and "0y"/"+1y" (annual). Annual rows are preferred. Quarterly rows are annualized ×4 and used as fallback. Estimates are rejected if < 90% of TTM (timing artefact guard).
- **DCF scenario model — mid-term growth cap:** `base_mid` is capped at 35% to prevent cyclical recovery spikes (e.g., MU 196% 2Y CAGR) from producing astronomical 5-year revenues. See `SCENARIO_MODEL_LIMITATIONS.md` for full list of 12 model limitations.
- **DCF scenario model — bull ordering guarantee:** `bull_mid` is floored at `base_mid × 1.20` to ensure bear < base < bull ordering when yfinance returns fewer than 8 quarters of quarterly income data (needed to compute historical growth percentiles).
- **DCF scenario model — INTC (and similar fab-heavy companies):** INTC correctly classifies as `is_cyclical_hardware=True` and receives a 10× terminal multiple cap. However, all three scenario prices return `None` because DCF equity = PV(FCF) + net_cash is negative in all scenarios: INTC's capex is ~28% of revenue (massive fab investment cycle), SBC ~5%, and opex ~13%, producing FCF margin ≈ −8% in year 1. Combined with ~−$70B net debt, the equity value floors below zero. This does not trigger the `deeply_negative_fcf` guard (threshold: FCF margin < −50%) — it is a separate path in `_dcf()` that returns `None` when `equity <= 0`. None prices for INTC are correct model behavior, not a bug. Use EV/Sales or P/Book for INTC valuation while it remains in its fab buildout cycle.
- **DCF scenario model — international ADRs (e.g. TSM, ASML, SAP):** yfinance returns inconsistent price vs revenue currency for non-US ADRs — share price may be in USD while revenue figures are in local currency (TWD for TSM, EUR for ASML/SAP). This produces wildly inflated or deflated DCF implied prices. Do not rely on scenario analysis output for non-US ADRs without verifying currency consistency. This is a yfinance data limitation, not a model bug.
- **Google Trends rate limiting** — pytrends returns 400 when hit too frequently. Circuit breaker opens after 3 errors in 10 min and skips Trends for the window. Sentiment model and sentiment validation fail silently with an error state.
- **yfinance `earnings_history`** — only returns last 4 quarters. Validation uses `earnings_dates` instead (up to 24+ quarters).
- **Factor model validation** — the factor model rarely passes validation for US equities. This is statistically expected: lagged weekly factor returns have very weak predictive power for next-week stock returns. The model is useful for attribution, not directional prediction.
- **Port conflicts** — if you see "Address already in use", run `lsof -ti:8000 | xargs kill -9`.
- **Adding a new tool** — use `_tool_schema(name, description, example)` from `tools_base` instead of a raw dict. Import `from tools_base import _tool_schema` at the top of the sector file. For non-ticker inputs (e.g., sector string), write the dict manually — `_tool_schema` is for single-ticker tools only.
- **Cache TTLs** — snapshot data: 30 min. DCF core: 4h. Behavioral layer: uncached (fresh each call). Validation: 6h. Flush scenario cache: `DELETE FROM cache WHERE key LIKE 'pred_scenario%'` via sqlite3.

---

## Changes — 2026-04-03 (refactor session)

### Utility extraction and boilerplate compression

**`tools_base.py`** (new) — shared helpers imported by every `tools_*.py` file. Must stay pure Python — no API clients (Anthropic client stays in `tools_universal.py`):
- `_tool_schema(name, description, example)` — generates the standard single-ticker tool definition dict. All 19 tool definitions across 8 files were converted to use this. **Always use this when adding a new single-ticker tool; never write the dict literal by hand.**
- `fetch_info(ticker_or_obj, extra_fields)`, `safe_ratio`, `_find_series`, `tool_result`

**`data_sources/_http.py`** (new) — shared HTTP helpers:
- `get(url, *, headers, timeout, **kwargs)` — wraps `requests.get` with default `User-Agent` header and `raise_for_status()`
- `get_cached_http(cache_key_str, url, ...)` — fetch + JSON parse + cache in one call
- Note: `stocktwits_sentiment.py` does not use `_http.get` because it handles 404 specially (ticker-not-found is a valid state, not an error). `sec_edgar.py` has its own `_get()` helper with EDGAR-specific rate limiting and headers.

**`predictive/` package** (new) — low-level helpers extracted from `predictive_analytics.py`:
- `predictive/_timeseries.py`: `align_weekly` (fetch + TZ-strip + resample to W-FRI), `robust_sigma` (MAD-based annualized σ), `zscore_log`, `decay_weights(n, lam=0.98)`
- `predictive/_ridge.py`: `weighted_ridge(X, y, weights)` — GCV-tuned Ridge with sandwich SEs

**`predictive_analytics.py` aliases** — the extracted functions are exposed internally as aliases so existing call sites don't break:
```python
_fetch_weekly = align_weekly   # backward-compat alias
_ols = weighted_ridge          # backward-compat alias
```
`align_weekly` does TZ normalization + resample internally, so the normalization loop in `get_factor_attribution` (lines ~109–117) is now a no-op on already-normalized data. It is harmless but redundant — can be removed in a future cleanup pass.

**`validation.py` import rule** — imports `_fetch_weekly`, `_ols`, and `decay_weights` exclusively from the `predictive.*` submodules, not from `predictive_analytics`. This is intentional and must not be reverted:
```python
from predictive._timeseries import align_weekly as _fetch_weekly, decay_weights as _decay_weights
from predictive._ridge import weighted_ridge as _ols
from predictive_analytics import SECTOR_ETF_MAP, FACTOR_LABELS  # constants only
```

**Function aliases added to data sources** (for a consistent import path):
- `data_sources/sec_edgar.py`: `get_insider_activity = get_edgar_form4`
- `data_sources/fred_macro.py`: `get_macro_context = get_fred_macro`
- Existing callers (`master_signal.py`, `server.py`) continue using the original names — both names point to the same function.

**Deleted:** `copy_of_isom260_session4_build_your_first_agent (1).py` — stale course exercise file, no longer needed.

---

## Changes — 2026-04-02 to 2026-04-03

### Scenario Analysis — complete DCF rebuild (`predictive_analytics.py`)

Replaced the EV/Revenue statistical percentile model with a three-stage fundamental DCF. This is the largest change in the codebase to date.

**New function structure:**
- `_compute_dcf_core(ticker, peer_medians)` — pure fundamentals, cached 4h
- `_apply_behavioral(dcf, behavioral_inputs)` — probability/narrative adjustments, always fresh
- `get_scenario_analysis(ticker, behavioral_inputs, peer_medians)` — public wrapper managing the cache split

**Revenue estimate parsing fix:** yfinance `revenue_estimate` index mixes "0q"/"+1q" (quarterly) and "0y"/"+1y" (annual) rows. Previous code used `rows[0]` as fallback which always resolved to the quarterly row. Fixed by explicitly separating annual rows (`"y" in r and "q" not in r`) and quarterly rows, preferring annual, annualizing quarterly ×4. This changed NVDA's Y1 estimate from $78.7B (1 quarter) to $369.4B (annual).

**2Y revenue CAGR fix:** Replaced yfinance's `revenueGrowth` field (MU showed 196.3% cyclical recovery artifact) with a 2Y CAGR computed from quarterly income statement: `(ttm_now / ttm_2y) ^ 0.5 − 1`. Falls back to quarter-over-quarter if fewer than 8 quarters available.

**Bugs fixed in DCF:**
- `base_mid` cap: consensus-implied Y1→Y2 growth (e.g., MU 52%) was flowing through uncapped to years 3–5; now capped at 35%
- `bull_mid` floor: `max(base_mid × 1.20, hist_p75_growth)` guarantees bull > base when yfinance returns <8 quarters of quarterly data
- `base_driver` text: was showing `rev_cagr_2y` even when `using_consensus=True`; now shows consensus Y1 estimate and near-term growth vs TTM

**Guards added:**
- `is_financial`: banks/insurers (grossMargins ≈ 0) → `dcf_not_applicable = True`, prices return None
- `deeply_negative_fcf`: FCF margin yr1 < −50% → `dcf_not_applicable = True`, prices return None

**Test results (2026-04-03, neutral behavioral inputs):**
| Ticker | Current | Bear | Base | Bull | Notes |
|--------|---------|------|------|------|-------|
| NVDA | $177 | $125 | $420 | $757 | bear < base < bull ✓ |
| MU | $366 | $224 | $1304 | $3093 | bear < base < bull ✓; high due to $108B Y1 consensus |
| QBTS | $14 | None | None | None | deeply_negative_fcf flag ✓ |
| JPM | $295 | None | None | None | financial company flag ✓ |

### Factor Attribution upgrade (`predictive_analytics.py`)
- Added 2 new factors: 1-Week Reversal (contrarian signal), Value/Growth Spread (IWD/IWF ratio)
- Upgraded OLS → Weighted Ridge regression with GCV cross-validated L2 penalty
- Sandwich covariance for SEs, weighted R²

### Server restructuring (`server.py`)
- `master_signal` now runs before `run_all_predictive` in Step 5 to populate `behavioral_inputs`
- `divergence_score` convention flip: master_signal HIGH = undervalued → `100 − divergence_score` passed to DCF (spec: HIGH = overhyped)
- `insider_signal` mapped from `insider_score` integer to string category (strongly_bullish/bullish/neutral/bearish/strongly_bearish)
- Performance breakdown tracking (`_perf` list) added throughout the SSE handler

### Dashboard (`dashboard.html`)
- **ScenarioRange component:** Added FCF year 1–3 table per card, model inputs row (discount rate, terminal multiple, peer median EV/Rev), flags row (yellow warning boxes for DCF-not-applicable cases), narrative adjustment line on base card showing pure DCF price alongside adjusted price, limitations panel (collapsed by default, expandable)
- **PredictTab:** Probability sliders now seeded from model-computed probabilities on first load (`weightsCustomized` state tracks whether user has overridden them; model re-seeds only if untouched)

### Validation (`validation.py`)
- Added `validate_scenario_model(ticker)` — quarterly walk-forward backtest comparing probability-weighted DCF target to actual price 52 weeks later; coverage_rate ≥ 65% over ≥ 6 periods = validated

### Data sources
- `data_sources/open_insider.py` deleted — consistently hit the 8s timeout; insider signals now from SEC EDGAR Form 4 only
- `SCENARIO_MODEL_LIMITATIONS.md` created — documents 12 known limitations of the DCF model with bias direction, severity rating, and what would fix each

---

## Performance (as of 2026-04-03)

**Cold start** (no cache): ~22-28s total (step1 0.4s + parallel tools 2.5s + master_signal 11-16s + Claude synthesis 8-12s)
**Warm run** (SQLite cache): ~10-14s total (most sources instant from cache; bottleneck = Claude synthesis)

**Dependency graph after optimization:**
```
get_sector_profile (0.4s, required first)
┌─ get_sector_behavioral_biases (0ms, instant)
├─ get_stock_price, get_company_info, get_financial_data ...  ← all tools in parallel
└─ get_reddit_sentiment (bottleneck ~2.5s)
→ wall clock: ~2.5s (was 8-12s sequential)

master_signal [8 sources, parallel] → behavioral_inputs → run_all_predictive
  Phase 1 [parallel]: factor attribution, earnings probability, sentiment
  Phase 2: scenario analysis (DCF core cached 4h; behavioral layer always fresh)
→ wall clock: ~11-16s

Claude synthesis → ~8-12s
```

**Profile run: `python3 profile_analysis.py NVDA`**

## What's Next (potential future work)

- [ ] Sentiment validation test with working Google Trends access
- [ ] Factor model with level-based predictors (vs. return-based) for better directional accuracy
- [ ] CI error bars rendered on the Factor Attribution chart in the dashboard
- [ ] Vercel/cloud deployment (requires porting backend from Python to Next.js API routes, or using a Python hosting service like Railway/Render)
- [ ] Additional sectors: Basic Materials, Utilities
- [ ] Reddit sentiment with live PRAW auth (currently falls back to public API)
- [ ] DCF model improvements: historical balance sheet data (limitation #1), NTM revenue estimates (limitation #2), regime-break detection for σ (limitation #4) — all require paid data provider
