# Stock Research Agent

An AI-powered stock research tool for IT sector analysis, built with the Claude API and free data sources.

## Features

- **8 research tools** — yfinance fundamentals, Yahoo Finance RSS news, OpenInsider filings, Reddit sentiment, and more
- **Behavioral finance layer** — narrative vs. reality analysis, insider signal scoring
- **Sector-specific tools** — tailored analysis for tech, healthcare, financials, industrials, consumer, energy, and real estate
- **Predictive analytics** — risk decomposition, signal aggregation
- **Web dashboard** — FastAPI backend with JWT auth and daily usage quotas

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
ADMIN_EMAIL=your_email_here
```

## Usage

Run the research agent directly:

```bash
python stock_research_agent.py
```

Start the API server (dashboard):

```bash
uvicorn server:app --reload
```

## Project Structure

| File | Purpose |
|------|---------|
| `stock_research_agent.py` | Main agent — 8-tool Claude-powered research loop |
| `stock_agent.py` | Earlier 4-tool version |
| `server.py` | FastAPI server with auth and quota management |
| `dashboard.html` | Frontend UI |
| `auth.py` | JWT authentication |
| `agent_core.py` | Core agent utilities |
| `master_signal.py` | Signal aggregation logic |
| `predictive_analytics.py` | ML-based predictive models |
| `risk_decomposition.py` | Risk factor analysis |
| `tools_*.py` | Sector-specific tool modules |
| `validation.py` | Data validation layer |

## Data Sources

All free, no paid API keys required (beyond Anthropic):

- **yfinance** — price data, fundamentals, earnings
- **Yahoo Finance RSS** — news and press releases
- **OpenInsider** — SEC insider transaction filings
- **Reddit** — sentiment via PRAW
- **Google Trends** — search interest via pytrends

## Deployment

Configured for [Fly.io](https://fly.io) — see `fly.toml` and `Dockerfile`.
