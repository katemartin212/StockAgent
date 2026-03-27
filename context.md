# Stock Research Agent — Project Context

Last updated: 2026-03-27

---

## What This Is

A production-grade, multi-sector AI stock research terminal. It runs locally via a FastAPI backend (`server.py`) and a single-file React dashboard (`dashboard.html`). The AI layer uses the Anthropic Claude API (`claude-sonnet-4-6`) to synthesize live data from 8+ free data sources into an institutional-grade research report.

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
| `dashboard.html` | Single-file React + Chart.js frontend (~3000 lines) |
| `master_signal.py` | Aggregates all data sources into a master composite signal |
| `stock_research_agent.py` | Main agent entry point (8-tool version, latest) |
| `stock_agent.py` | Earlier 4-tool version (preserved) |

### Predictive Analytics (Predict Tab)
| File | Purpose |
|------|---------|
| `predictive_analytics.py` | 4 forward-looking models: factor attribution, earnings probability, scenario analysis, sentiment mean reversion |
| `validation.py` | Walk-forward out-of-sample validation framework. Produces HIGH/MEDIUM/LOW/UNVALIDATED tier |

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
| `_cache.py` | In-memory TTL cache (30min default, 4h predictive, 6h validation) |
| `sec_edgar.py` | SEC EDGAR free API — financial filings |
| `open_insider.py` | OpenInsider — Form 4 insider transactions |
| `reddit_sentiment.py` | Reddit API (PRAW) — retail sentiment with relevance filtering |
| `stocktwits_sentiment.py` | StockTwits API — trader sentiment |
| `trends_signal.py` | Google Trends via pytrends |
| `fred_macro.py` | FRED St. Louis Fed — macro indicators |
| `comps_data.py` | yfinance peer comparison table |

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
- Factor Attribution — OLS bar chart with FDR-corrected significance, 95% CIs, current headwinds/tailwinds
- Earnings Surprise Probability — semicircle gauge with bootstrap 95% CI, sub-scores, 8Q history chart
- Scenario Analysis — bear/base/bull range with live probability sliders
- Sentiment Mean Reversion — z-score line chart with ±1.5σ bands, correlation validation

---

## Predictive Models & Validation

### `predictive_analytics.py` — 4 models

**1. Factor Attribution** (`get_factor_attribution`)
- 6-factor OLS: 10Y yield, DXY, VIX, sector ETF, S&P 500, 4W momentum
- Weekly returns, trailing 2 years
- Benjamini-Hochberg FDR correction on all p-values
- 95% CI on each coefficient
- Current conditions vs. prior-history percentile (look-ahead bias fixed: uses `iloc[:-1]`)

**2. Earnings Surprise Probability** (`get_earnings_surprise_probability`)
- 4 sub-scores: historical beat rate (35%), revision momentum (35%), analyst sentiment (15%), sector read-through (15%)
- Filters to past-only events (look-ahead bias fixed)
- Bootstrap 95% CI on final probability (1000 resamples, seed=42)

**3. Scenario Analysis** (`get_scenario_analysis`)
- Bear/base/bull using 2Y EV/Revenue percentile range
- EV/Rev percentile excludes last 13 weeks (look-ahead bias fixed)
- Live probability sliders recalculate weighted target in real time

**4. Sentiment Mean Reversion** (`get_sentiment_mean_reversion`)
- 2Y Google Trends weekly z-score vs. 26W baseline
- Baseline uses `iloc[:-2]` to exclude incomplete recent weeks (look-ahead bias fixed)
- Correlation with 4W forward returns

### `validation.py` — 3 validation tests

**Factor Model validation** (`validate_factor_model`)
- Lagged walk-forward OLS (X[t] predicts y[t+1])
- 5Y weekly data, 52W minimum training window
- Validated if: directional accuracy > 52% AND |IC| > 0.04
- Key bug fixed: mixed timezones (Chicago vs NY) caused 2× row duplication — all series now resampled to W-FRI UTC

**Earnings Model validation** (`validate_earnings_model`)
- Leave-one-out CV on `earnings_dates` (up to 24+ historical events)
- Brier Score and Brier Skill Score vs. 50% naive baseline
- Validated if: brier_skill > 0 AND n_events ≥ 6

**Sentiment Signal validation** (`validate_sentiment_signal`)
- t-test on forward returns conditioned on |z| > 1.5 signals
- Bootstrap 95% CI on mean forward return (1000 resamples)
- Validated if: |t-stat| > 1.96 AND n_signals ≥ 8

**Confidence Tiers:**
- HIGH = 3/3 validated
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

- **Google Trends rate limiting** — pytrends returns 400 when hit too frequently. Sentiment model and sentiment validation fail silently with an error state. Wait a few minutes and retry.
- **yfinance `earnings_history`** — only returns last 4 quarters. Validation uses `earnings_dates` instead (up to 24+ quarters).
- **Factor model validation** — the factor model rarely passes validation for US equities. This is statistically expected: lagged weekly factor returns have very weak predictive power for next-week stock returns. The model is useful for attribution, not directional prediction.
- **Port conflicts** — if you see "Address already in use", run `lsof -ti:8000 | xargs kill -9`.
- **Cache TTLs** — snapshot data: 30 min. Predictive data: 4h. Validation: 6h. Clear cache by restarting the server process.

---

## What's Next (potential future work)

- [ ] Sentiment validation test with working Google Trends access
- [ ] Factor model with level-based predictors (vs. return-based) for better directional accuracy
- [ ] CI error bars rendered on the Factor Attribution chart in the dashboard
- [ ] Vercel/cloud deployment (requires porting backend from Python to Next.js API routes, or using a Python hosting service like Railway/Render)
- [ ] Additional sectors: Basic Materials, Utilities
- [ ] Reddit sentiment with live PRAW auth (currently falls back to public API)
