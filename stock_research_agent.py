#!/usr/bin/env python3
"""
stock_research_agent.py
-----------------------
Production-grade IT sector stock research agent with behavioral finance analysis.

11 tools:
  1. calculator                — precise arithmetic
  2. get_stock_price           — live price + trading metrics (yfinance)
  3. get_company_info          — sector, employees, CEO, HQ (yfinance)
  4. get_financial_data        — Rule of 40, margins, valuation multiples (yfinance)
  5. get_news_sentiment        — headlines + bullish/bearish score (Yahoo Finance RSS)
  6. get_insider_trades        — Form 4 buy/sell activity (OpenInsider, free scrape)
  7. get_earnings_surprise     — EPS beat/miss history (yfinance)
  8. get_reddit_sentiment      — r/wallstreetbets + r/investing post volume & tone
  9. get_net_revenue_retention — NRR proxy via deferred revenue + same-quarter YoY (yfinance)
 10. get_dcf_implied_growth    — reverse-DCF: what revenue CAGR does the stock price imply?
 11. get_dilution_rate         — SBC as % revenue/gross profit, SBC-adjusted FCF margin, share count growth

All data sources are free and require no API key.
Flagged below anywhere a free signup would unlock more data.

Setup:
    pip install -r requirements.txt
    Add ANTHROPIC_API_KEY to .env or export it in your shell.
"""

import os
import re
import json
import time
from datetime import datetime

import pandas as pd
import requests
import feedparser
import yfinance as yf
from bs4 import BeautifulSoup
import anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── API client ────────────────────────────────────────────────────────────────

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set.\n"
        "Add it to a .env file: ANTHROPIC_API_KEY=sk-ant-...\n"
        "Or export it: export ANTHROPIC_API_KEY=sk-ant-..."
    )

client = anthropic.Anthropic(api_key=api_key)
print("✅ Anthropic client ready")

# ── System prompt ─────────────────────────────────────────────────────────────

_today = datetime.now().strftime("%B %d, %Y")

SYSTEM_PROMPT = f"""You are a senior IT sector equity analyst at a quantitative investment fund. \
Today's date is {_today}. You have access to 11 tools covering fundamentals, valuation, \
dilution, revenue retention, news, insider activity, earnings history, and social sentiment.

When analyzing a stock or comparing stocks:
1. Call all relevant tools before drawing conclusions — never guess at data.
2. Use the calculator for any derived metric (market cap, shares to buy, growth math).
3. Always produce output in the structured report format below.

Tool usage rules:
- ALWAYS run get_dcf_implied_growth and get_dilution_rate whenever the question involves \
valuation, price targets, or whether a stock is cheap or expensive. These make the market's \
embedded assumptions explicit and surface the true cost of equity compensation.
- ALWAYS run get_net_revenue_retention for SaaS, cloud, or subscription businesses \
(e.g. CRM, NOW, SNOW, DDOG, ZS, OKTA, MSFT, ADBE). Skip it for hardware, semiconductors, \
and non-recurring revenue models — the tool will tell you this automatically.
- get_dcf_implied_growth is the single most important valuation context tool: it translates \
an abstract stock price into a concrete growth assumption that can be compared against history.
- get_dilution_rate reveals the true FCF margin after accounting for stock-based compensation \
— always include it when retail investors are citing GAAP earnings or reported FCF.

──────────────────────────────────────────────────────────────
REPORT FORMAT  (use this exact structure for every analysis)
──────────────────────────────────────────────────────────────

# 📊 {{TICKER}} — {{Company Name}}
*{_today}*

---

## 1. Fundamentals
- **Price / Market Cap / P/E / Forward P/E / EV-Revenue**
- **Revenue Growth | Gross Margin | FCF Margin | Rule of 40**
- **Analyst Consensus | Price Target**

## 2. Earnings Track Record (Last 4–6 Quarters)
- Beat rate, miss count, average EPS surprise %

## 3. News Sentiment (Last 7 Days)
- Overall: Bullish / Bearish / Neutral  |  Score: X / 100
- Key headlines (3–5 bullets)

## 4. Social Sentiment — Reddit (Last 7 Days)
- Posts: N (WSB: X, r/investing: Y)  |  Sentiment: ...
- Top community narratives

## 5. Insider Activity (Last 90 Days)
- Purchases: N  |  Sales: N  |  Net signal: Bullish / Bearish / Neutral
- Notable transactions

---

## 🎯 Narrative vs Reality

**🗣️ Retail Narrative:** What retail investors appear to believe \
(synthesized from news + Reddit tone)

**📊 Fundamental Reality:** What the hard data actually shows \
(Rule of 40, margins, earnings beats, valuation)

**⚡ Divergence Score: X / 100**
> 0–20 Aligned | 21–40 Mild gap | 41–60 Notable | 61–80 Strong | 81–100 Extreme

**🏷️ Recommended Action:** ACCUMULATE / HOLD / TRIM / AVOID

**💡 Rationale:** 2–3 sentences grounded in the divergence between perception and reality.

---
*Sources: Yahoo Finance (live) · OpenInsider (free) · Reddit public API*
"""

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Calculator ─────────────────────────────────────────────────────────────

calculator_tool = {
    "name": "calculator",
    "description": (
        "Precise arithmetic for financial math: +, -, *, /, ** and parentheses. "
        "Use this for every numerical calculation — never do mental math."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "e.g. '242.58 * 15120000000' or '(385 / 164000) * 1000000'"
            }
        },
        "required": ["expression"]
    }
}

def calculator(expression: str) -> str:
    try:
        if not all(c in "0123456789+-*/.() " for c in expression):
            return "Error: only numbers and +-*/.() are allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# ── 2. Stock Price (live via yfinance) ────────────────────────────────────────

stock_price_tool = {
    "name": "get_stock_price",
    "description": (
        "Live stock price, daily change, 52-week range, volume, and market cap. "
        "Use get_financial_data instead when the question involves valuation metrics."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'"}
        },
        "required": ["ticker"]
    }
}

def get_stock_price(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker.upper().strip()).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return json.dumps({"error": f"No price data found for '{ticker}'."})
        return json.dumps({
            "ticker": ticker.upper(),
            "name": info.get("longName"),
            "price": price,
            # regularMarketChangePercent is already in percent form (e.g. -0.39 = -0.39%)
            "change_pct_today": round(info.get("regularMarketChangePercent") or 0, 2),
            "high_52w": info.get("fiftyTwoWeekHigh"),
            "low_52w": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume"),
            "avg_volume_30d": info.get("averageVolume"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "market_cap": info.get("marketCap"),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── 3. Company Info (live via yfinance) ───────────────────────────────────────

company_info_tool = {
    "name": "get_company_info",
    "description": (
        "Company profile: sector, industry, employee count, CEO, headquarters, "
        "and a brief business description. Use ticker symbol as input."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'MSFT'"}
        },
        "required": ["ticker"]
    }
}

def get_company_info(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker.upper().strip()).info
        officers = info.get("companyOfficers") or []
        # Find CEO by title; fall back to first officer listed
        ceo = next(
            (o.get("name") for o in officers if "CEO" in (o.get("title") or "").upper()),
            officers[0].get("name") if officers else "N/A"
        )
        city = info.get("city") or ""
        state = info.get("state") or ""
        hq = ", ".join(filter(None, [city, state]))
        description = info.get("longBusinessSummary") or ""
        return json.dumps({
            "ticker": ticker.upper(),
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "employees": info.get("fullTimeEmployees"),
            "ceo": ceo,
            "hq": hq,
            "description": description[:400] + ("..." if len(description) > 400 else ""),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── 4. Financial Data (live via yfinance) ─────────────────────────────────────

financial_data_tool = {
    "name": "get_financial_data",
    "description": (
        "Institutional-grade financial metrics: revenue growth, gross margin, FCF margin, "
        "Rule of 40 (= revenue growth % + FCF margin %; >40 is healthy for tech), "
        "EV/Revenue, P/E, forward P/E, and analyst price targets. "
        "Use this whenever the question involves valuation, financial health, or peer comparison."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'NVDA'"}
        },
        "required": ["ticker"]
    }
}

def get_financial_data(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not info or price is None:
            return json.dumps({"error": f"No data for '{ticker}'."})

        # Revenue growth (YoY, from annual income statement)
        revenue_growth = None
        try:
            revenues = stock.financials.loc["Total Revenue"].dropna()
            if len(revenues) >= 2:
                revenue_growth = round(
                    ((revenues.iloc[0] - revenues.iloc[1]) / abs(revenues.iloc[1])) * 100, 1
                )
        except Exception:
            pass

        # FCF margin (Free Cash Flow / Total Revenue)
        fcf_margin = None
        try:
            fcf = stock.cashflow.loc["Free Cash Flow"].iloc[0]
            rev = info.get("totalRevenue") or 1
            fcf_margin = round((fcf / rev) * 100, 1)
        except Exception:
            pass

        # Rule of 40
        rule_of_40 = (
            round(revenue_growth + fcf_margin, 1)
            if revenue_growth is not None and fcf_margin is not None
            else "N/A"
        )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "price": price,
            "market_cap": info.get("marketCap"),
            "pe_ratio": round(info.get("trailingPE") or 0, 1),
            "forward_pe": round(info.get("forwardPE") or 0, 1),
            "ev_revenue_multiple": round(info.get("enterpriseToRevenue") or 0, 1),
            "revenue_growth_pct": revenue_growth,
            "gross_margin_pct": round((info.get("grossMargins") or 0) * 100, 1),
            "fcf_margin_pct": fcf_margin,
            "rule_of_40": rule_of_40,
            "short_interest_pct": round((info.get("shortPercentOfFloat") or 0) * 100, 1),
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_recommendation": (info.get("recommendationKey") or "N/A").replace("_", " "),
            "data_source": "Yahoo Finance (live)",
        })
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"})


# ── 5. News Sentiment ─────────────────────────────────────────────────────────
# Source: Yahoo Finance RSS — free, no API key required.
# 💡 For broader coverage (100+ sources, 30-day history): sign up for a free
#    key at newsapi.org and replace the RSS URL with the NewsAPI endpoint.

news_sentiment_tool = {
    "name": "get_news_sentiment",
    "description": (
        "Fetch recent news headlines for a ticker from Yahoo Finance RSS and score "
        "them bullish/bearish/neutral via keyword analysis. Returns an overall sentiment "
        "label, a score from -100 (very bearish) to +100 (very bullish), and the top "
        "headlines. Use this to understand the current media narrative around a stock."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'TSLA'"}
        },
        "required": ["ticker"]
    }
}

_NEWS_BULLISH = {
    "beat", "surge", "rally", "upgrade", "buy", "strong", "growth", "record",
    "profit", "breakthrough", "bullish", "outperform", "raises", "soars",
    "jumps", "climbs", "boosts", "wins", "exceeds", "better", "positive",
}
_NEWS_BEARISH = {
    "miss", "fall", "decline", "downgrade", "sell", "weak", "loss", "concern",
    "risk", "cut", "bearish", "underperform", "lowers", "drops", "plunges",
    "slides", "disappoints", "worse", "below", "warning", "layoffs", "lawsuit",
}

def get_news_sentiment(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:15]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")[:200]
            words = set(re.findall(r"\b\w+\b", (title + " " + summary).lower()))
            bull_hits = words & _NEWS_BULLISH
            bear_hits = words & _NEWS_BEARISH
            sentiment = (
                "bullish" if len(bull_hits) > len(bear_hits)
                else "bearish" if len(bear_hits) > len(bull_hits)
                else "neutral"
            )
            articles.append({
                "title": title,
                "published": entry.get("published", ""),
                "sentiment": sentiment,
                "signals": {
                    "bullish": sorted(bull_hits),
                    "bearish": sorted(bear_hits),
                },
            })

        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for a in articles:
            counts[a["sentiment"]] += 1

        total = len(articles) or 1
        score = round((counts["bullish"] - counts["bearish"]) / total * 100, 1)
        overall = "bullish" if score > 10 else "bearish" if score < -10 else "neutral"

        return json.dumps({
            "ticker": ticker,
            "articles_analyzed": len(articles),
            "overall_sentiment": overall,
            "sentiment_score": score,          # -100 to +100
            "breakdown": counts,
            "top_headlines": [
                {"title": a["title"], "sentiment": a["sentiment"], "published": a["published"]}
                for a in articles[:5]
            ],
            "data_source": "Yahoo Finance RSS (free, no key required)",
            "upgrade_note": "💡 newsapi.org free tier gives broader coverage across 100+ sources",
        })
    except Exception as e:
        return json.dumps({"error": f"News fetch failed: {e}"})


# ── 6. Insider Trades ─────────────────────────────────────────────────────────
# Source: OpenInsider (openinsider.com) — free, no API key required (scraped).
# OpenInsider aggregates SEC Form 4 filings in near real-time.

insider_trades_tool = {
    "name": "get_insider_trades",
    "description": (
        "Fetch recent Form 4 insider buying and selling for a stock from OpenInsider "
        "(last 90 days). Returns purchase/sale counts, net insider sentiment "
        "(bullish/bearish/neutral), and a list of recent transactions. "
        "Heavy purchasing by C-suite insiders is a strong bullish signal."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'META'"}
        },
        "required": ["ticker"]
    }
}

def get_insider_trades(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        # OpenInsider screener: last 90 days, purchases (xp=1) and sales (xs=1)
        url = (
            f"https://openinsider.com/screener?s={ticker}"
            "&fd=90&xp=1&xs=1&sortcol=0&cnt=20&Action=Submit"
        )
        headers = {"User-Agent": "Mozilla/5.0 StockResearchBot/1.0"}
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        table = soup.find("table", class_="tinytable")
        if not table:
            return json.dumps({
                "ticker": ticker,
                "trades_found": 0,
                "insider_sentiment": "neutral",
                "message": "No insider trades in last 90 days.",
                "data_source": "OpenInsider (free, no key required)",
            })

        # Column layout: [X, Filing Date, Trade Date, Ticker, Company,
        #                  Insider Name, Title, Trade Type, Price, Qty,
        #                  Owned, ΔOwn, Value]
        trades = []
        for row in table.find_all("tr")[1:]:   # skip header row
            cols = row.find_all("td")
            if len(cols) < 13:
                continue
            trades.append({
                "filing_date": cols[1].text.strip(),
                "trade_date": cols[2].text.strip(),
                "insider": cols[5].text.strip(),
                "title": cols[6].text.strip(),
                "trade_type": cols[7].text.strip(),
                "price": cols[8].text.strip(),
                "quantity": cols[9].text.strip(),
                "value": cols[12].text.strip(),
            })

        # "P - Purchase" = open-market buy (most significant signal)
        # "S - Sale" = open-market sale  |  "A - Award" = stock grant (ignore for signal)
        purchases = [t for t in trades if "P - Purchase" in t["trade_type"]]
        sales = [t for t in trades if "S - Sale" in t["trade_type"] and "OE" not in t["trade_type"]]
        awards = [t for t in trades if "A - Award" in t["trade_type"]]

        net = len(purchases) - len(sales)
        sentiment = "bullish" if net > 0 else "bearish" if net < 0 else "neutral"

        return json.dumps({
            "ticker": ticker,
            "trades_found": len(trades),
            "purchases": len(purchases),
            "sales": len(sales),
            "awards_grants": len(awards),
            "insider_sentiment": sentiment,
            "recent_trades": trades[:8],
            "data_source": "OpenInsider (free, no key required)",
        })
    except Exception as e:
        return json.dumps({"error": f"Insider data fetch failed: {e}"})


# ── 7. Earnings Surprise ──────────────────────────────────────────────────────
# Source: Yahoo Finance via yfinance — free, no key required.

earnings_surprise_tool = {
    "name": "get_earnings_surprise",
    "description": (
        "Last 4–6 quarters of EPS estimate vs actual to show the beat/miss history. "
        "Returns beat rate %, average EPS surprise %, and per-quarter detail. "
        "Companies beating >75% of estimates command valuation premiums. "
        "Use this to assess earnings quality and forecast credibility."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'MSFT'"}
        },
        "required": ["ticker"]
    }
}

def get_earnings_surprise(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        df = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if df is None or df.empty:
            return json.dumps({"error": f"No earnings data found for '{ticker}'."})

        records = []
        for date, row in df.iterrows():
            actual_raw = row.get("Reported EPS")
            # Skip future quarters — Reported EPS is NaN until reported
            if pd.isna(actual_raw):
                continue
            try:
                estimate = float(row.get("EPS Estimate")) if not pd.isna(row.get("EPS Estimate")) else None
                actual = float(actual_raw)
                surprise = float(row.get("Surprise(%)")) if not pd.isna(row.get("Surprise(%)")) else None
            except (TypeError, ValueError):
                continue

            records.append({
                "date": str(date.date()),
                "eps_estimate": round(estimate, 3) if estimate is not None else None,
                "eps_actual": round(actual, 3),
                "surprise_pct": round(surprise, 2) if surprise is not None else None,
                "result": "beat" if (surprise is not None and surprise > 0) else "miss",
            })

        if not records:
            return json.dumps({"error": f"No reported quarters found for '{ticker}'."})

        beats = sum(1 for r in records if r["result"] == "beat")
        surprises = [r["surprise_pct"] for r in records if r["surprise_pct"] is not None]
        avg_surprise = round(sum(surprises) / len(surprises), 2) if surprises else None

        return json.dumps({
            "ticker": ticker,
            "quarters_analyzed": len(records),
            "beat_count": beats,
            "miss_count": len(records) - beats,
            "beat_rate_pct": round(beats / len(records) * 100, 1),
            "avg_eps_surprise_pct": avg_surprise,
            "history": records,
            "data_source": "Yahoo Finance via yfinance",
        })
    except Exception as e:
        return json.dumps({"error": f"Earnings data failed: {e}"})


# ── 8. Reddit Sentiment ───────────────────────────────────────────────────────
# Source: Reddit public JSON API — free, no account required.
# 💡 For higher rate limits: register a free app at reddit.com/prefs/apps
#    and use PRAW (pip install praw) with OAuth.

reddit_sentiment_tool = {
    "name": "get_reddit_sentiment",
    "description": (
        "Search r/wallstreetbets and r/investing for posts mentioning a ticker in the "
        "last 7 days. Returns post volume, engagement, overall tone "
        "(bullish/bearish/neutral via keyword analysis), and top posts. "
        "High WSB volume + bullish tone often signals retail FOMO and crowding risk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'TSLA'"}
        },
        "required": ["ticker"]
    }
}

_WSB_BULLISH = {
    "moon", "rocket", "bull", "buy", "calls", "long", "squeeze", "breakout",
    "yolo", "bullish", "dip", "accumulate", "undervalued", "upside",
}
_WSB_BEARISH = {
    "puts", "short", "sell", "crash", "dump", "bear", "overvalued", "drop",
    "fall", "bearish", "bubble", "avoid", "trim", "correction", "overbought",
}

def get_reddit_sentiment(ticker: str) -> str:
    ticker = ticker.upper().strip()
    headers = {"User-Agent": f"StockResearchBot/1.0 (ticker={ticker})"}
    subreddits = ["wallstreetbets", "investing"]
    all_posts = []

    for sub in subreddits:
        try:
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={ticker}&sort=new&limit=50&t=week&restrict_sr=1"
            )
            resp = requests.get(url, headers=headers, timeout=12)
            resp.raise_for_status()
            for post in resp.json().get("data", {}).get("children", []):
                d = post.get("data", {})
                all_posts.append({
                    "subreddit": sub,
                    "title": d.get("title", ""),
                    "score": d.get("score", 0),
                    "comments": d.get("num_comments", 0),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                })
            time.sleep(1)   # respect Reddit's rate limit
        except Exception:
            pass            # silently skip if one subreddit is unreachable

    if not all_posts:
        return json.dumps({
            "ticker": ticker,
            "posts_found": 0,
            "overall_sentiment": "neutral",
            "message": "No recent posts found — low retail attention signal.",
            "data_source": "Reddit public JSON API",
        })

    # Score each post title with keyword matching
    sent_scores = []
    for post in all_posts:
        words = set(re.findall(r"\b\w+\b", post["title"].lower()))
        bull = len(words & _WSB_BULLISH)
        bear = len(words & _WSB_BEARISH)
        sent_scores.append(1 if bull > bear else -1 if bear > bull else 0)

    avg = sum(sent_scores) / len(sent_scores)
    overall = "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral"

    # Top posts by combined engagement (score + weighted comment count)
    top = sorted(all_posts, key=lambda p: p["score"] + p["comments"] * 2, reverse=True)[:5]

    return json.dumps({
        "ticker": ticker,
        "posts_found": len(all_posts),
        "r_wallstreetbets": sum(1 for p in all_posts if p["subreddit"] == "wallstreetbets"),
        "r_investing": sum(1 for p in all_posts if p["subreddit"] == "investing"),
        "overall_sentiment": overall,
        "sentiment_score": round(avg, 3),   # -1.0 (bearish) to +1.0 (bullish)
        "top_posts": top,
        "data_source": "Reddit public JSON API (free, no key required)",
        "upgrade_note": "💡 Register a free app at reddit.com/prefs/apps for higher rate limits via PRAW",
    })


# ── 9. Net Revenue Retention ──────────────────────────────────────────────────
# NRR measures how much revenue a company retains AND expands from its existing
# customer base year over year. Above 120% means existing customers alone can
# drive strong growth even with zero new customer acquisition. It's the single
# most important metric for assessing the quality of a SaaS business model.
# For companies that don't break it out, we proxy via same-quarter YoY revenue
# growth (valid when the majority of revenue is recurring) and deferred revenue
# trends on the balance sheet.

nrr_tool = {
    "name": "get_net_revenue_retention",
    "description": (
        "Estimate Net Revenue Retention (NRR) for subscription/SaaS companies. "
        "NRR = (prior revenue + expansion - churn) / prior revenue × 100. "
        "Proxied via same-quarter YoY revenue growth and deferred revenue trends. "
        "Returns NRR estimate, expanding/contracting trend, benchmark tier, and "
        "plain-English interpretation. Returns a clear explanation if NRR is not "
        "applicable (e.g. hardware, semiconductors, non-subscription models)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'CRM'"}
        },
        "required": ["ticker"]
    }
}

def get_net_revenue_retention(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        info = stock.info

        sector   = info.get("sector", "")
        industry = info.get("industry", "")
        summary  = (info.get("longBusinessSummary") or "").lower()

        sub_keywords = ["software", "saas", "cloud", "subscription", "platform",
                        "service", "recurring", "software-as"]
        is_subscription = any(k in (sector + industry).lower() for k in sub_keywords) \
                       or any(k in summary for k in sub_keywords)

        if not is_subscription:
            if "semiconductor" in industry.lower() or "hardware" in industry.lower():
                alt = "Design Win Rate and Customer Concentration"
            elif "retail" in industry.lower() or "consumer" in sector.lower():
                alt = "Same-Store Sales Growth (SSSG)"
            else:
                alt = "Customer Concentration and Revenue Cohort Analysis"
            return json.dumps({
                "ticker": ticker,
                "nrr_applicable": False,
                "sector": sector,
                "industry": industry,
                "explanation": (
                    f"{info.get('longName', ticker)} operates in {industry} ({sector}), "
                    "which is not a subscription-based model. NRR is not meaningful here."
                ),
                "recommended_metric_instead": alt,
                "data_source": "Yahoo Finance via yfinance",
            })

        results = {"ticker": ticker, "nrr_applicable": True,
                   "sector": sector, "industry": industry}

        # Same-quarter YoY revenue growth proxy (needs ≥5 quarters)
        try:
            qrev = stock.quarterly_financials.loc["Total Revenue"].dropna()
            if len(qrev) >= 5:
                yoy_growths = []
                for i in range(min(4, len(qrev) - 4)):
                    curr  = float(qrev.iloc[i])
                    prior = float(qrev.iloc[i + 4])
                    if prior != 0:
                        yoy_growths.append(round((curr - prior) / abs(prior) * 100, 1))
                if yoy_growths:
                    avg_yoy = round(sum(yoy_growths) / len(yoy_growths), 1)
                    nrr_est = round(100 + avg_yoy, 1)
                    trend   = "expanding" if yoy_growths[0] > yoy_growths[-1] else "contracting"
                    results.update({
                        "nrr_estimate_pct": nrr_est,
                        "method": "Same-quarter YoY revenue growth proxy (4-quarter avg)",
                        "yoy_growth_by_quarter": yoy_growths,
                        "trend": trend,
                    })
        except Exception:
            pass

        # Deferred revenue trend from balance sheet
        try:
            bs = stock.quarterly_balance_sheet
            for row_name in ["Deferred Revenue", "Deferred Revenue And Credits",
                             "Current Deferred Revenue"]:
                if row_name in bs.index:
                    dr = bs.loc[row_name].dropna()
                    if len(dr) >= 4:
                        vals = [float(v) for v in dr.iloc[:4]]
                        if vals[-1] != 0:
                            dr_chg = round((vals[0] - vals[-1]) / abs(vals[-1]) * 100, 1)
                            results["deferred_revenue_yoy_pct"] = dr_chg
                            results["deferred_revenue_signal"] = (
                                "positive" if dr_chg > 5 else
                                "flat"     if abs(dr_chg) <= 5 else "negative"
                            )
                    break
        except Exception:
            pass

        if "nrr_estimate_pct" not in results and "deferred_revenue_yoy_pct" not in results:
            return json.dumps({
                "ticker": ticker,
                "nrr_applicable": True,
                "error": "Insufficient historical data — need at least 5 quarters of revenue.",
                "data_source": "Yahoo Finance via yfinance",
            })

        nrr = results.get("nrr_estimate_pct")
        if nrr is not None:
            if nrr < 100:
                benchmark = "Below 100% — net churn. Customer base is shrinking. High risk."
                health    = "poor"
            elif nrr < 110:
                benchmark = "100–110% — stable, marginal expansion. Acceptable but not best-in-class."
                health    = "stable"
            elif nrr < 120:
                benchmark = "110–120% — healthy expansion. Existing customers are growing their spend."
                health    = "healthy"
            else:
                benchmark = "Above 120% — exceptional. Existing customers are a strong organic growth engine."
                health    = "exceptional"
            results["benchmark"] = benchmark
            results["health_rating"] = health
            direction = "growing" if nrr >= 100 else "shrinking"
            results["plain_english"] = (
                f"Based on revenue retention patterns, {info.get('longName', ticker)}'s "
                f"existing customer base is {direction} at an estimated "
                f"{abs(nrr - 100):.1f}% net rate. "
                f"This is {'above' if nrr >= 110 else 'below'} the 110% threshold that "
                "typically separates high-growth SaaS from commodity subscription businesses."
            )

        results["data_source"] = (
            "Yahoo Finance via yfinance (proxy — exact NRR requires 10-Q RPO disclosures)"
        )
        return json.dumps(results)

    except Exception as e:
        return json.dumps({"error": f"NRR calculation failed: {e}"})


# ── 10. DCF Implied Growth ─────────────────────────────────────────────────────
# Reverse-engineers the revenue CAGR embedded in the current stock price.
# This is the most important single valuation context tool: it makes the market's
# implicit growth assumption explicit and comparable against historical delivery.
# Assumptions: 10% discount rate, 3% terminal growth, 20× terminal FCF multiple,
# constant FCF margin at the current level. Requires positive FCF to run.

dcf_implied_tool = {
    "name": "get_dcf_implied_growth",
    "description": (
        "Reverse-engineer the 10-year revenue CAGR the current stock price is implying, "
        "using a DCF model with 10% discount rate, 20× terminal FCF multiple, and "
        "constant FCF margin. Returns the implied CAGR, the actual trailing 3-year CAGR, "
        "the gap between them, a verdict on achievability, and a plain-English summary. "
        "This is the single most important valuation context tool — always use it when "
        "assessing whether a stock is cheap or expensive."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'NVDA'"}
        },
        "required": ["ticker"]
    }
}

def get_dcf_implied_growth(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info

        price      = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap = info.get("marketCap")
        revenue    = info.get("totalRevenue")

        if not all([price, market_cap, revenue]) or revenue <= 0:
            return json.dumps({"error": f"Missing price/market cap/revenue data for '{ticker}'."})

        # Free cash flow (most recent annual)
        fcf = None
        try:
            fcf = float(stock.cashflow.loc["Free Cash Flow"].iloc[0])
        except Exception:
            pass

        if fcf is None or fcf <= 0:
            return json.dumps({
                "ticker": ticker,
                "name": info.get("longName"),
                "error": (
                    "Cannot run DCF: free cash flow is negative or unavailable. "
                    "Pre-profitability companies cannot be valued on a DCF basis."
                ),
                "note": "Use EV/Revenue or EV/Gross Profit multiples for pre-FCF companies.",
                "data_source": "Yahoo Finance via yfinance",
            })

        fcf_margin     = fcf / revenue   # keep as decimal
        discount_rate  = 0.10
        terminal_mult  = 20

        # DCF(cagr) = revenue × fcf_margin × [Σ_{t=1}^{10} x^t  +  terminal_mult × x^10]
        # where x = (1 + cagr) / (1 + discount_rate)
        # Geometric series: Σ x^t = x*(1 - x^10)/(1 - x)  for x ≠ 1
        def dcf_value(cagr: float) -> float:
            x = (1 + cagr) / (1 + discount_rate)
            if abs(x - 1) < 1e-9:
                pv_flows = 10.0
            else:
                pv_flows = x * (1 - x**10) / (1 - x)
            return revenue * fcf_margin * (pv_flows + terminal_mult * x**10)

        # Binary search over [-30%, +200%]
        lo, hi = -0.30, 2.00
        for _ in range(80):
            mid = (lo + hi) / 2
            if dcf_value(mid) < market_cap:
                lo = mid
            else:
                hi = mid
        implied_cagr = round(mid * 100, 1)

        # Trailing revenue CAGR (up to 3 years)
        trailing_cagr = None
        try:
            revs = stock.financials.loc["Total Revenue"].dropna()
            if len(revs) >= 4:
                trailing_cagr = round(
                    ((float(revs.iloc[0]) / float(revs.iloc[3])) ** (1/3) - 1) * 100, 1
                )
            elif len(revs) >= 2:
                n = len(revs) - 1
                trailing_cagr = round(
                    ((float(revs.iloc[0]) / float(revs.iloc[-1])) ** (1/n) - 1) * 100, 1
                )
        except Exception:
            pass

        gap = round(implied_cagr - trailing_cagr, 1) if trailing_cagr is not None else None

        if gap is None:
            verdict = "Insufficient historical data to compare against implied growth."
        elif gap <= 0:
            verdict = "Market prices in growth BELOW historical delivery — potential value opportunity if fundamentals hold."
        elif gap <= 5:
            verdict = "Market prices in growth roughly in line with history. Achievable if execution continues."
        elif gap <= 15:
            verdict = "Market prices in meaningfully above historical growth. Requires acceleration — possible but not certain."
        else:
            verdict = "Market prices in exceptional growth far above historical rates. Requires a step-change in scale or margin."

        plain = (
            f"To justify its ${price:.2f} share price, {info.get('longName', ticker)} "
            f"needs to grow revenue at {implied_cagr}% per year for 10 years. "
        )
        if trailing_cagr is not None:
            if gap is not None and abs(gap) <= 3:
                plain += f"It has historically grown at {trailing_cagr}% — roughly in line with what the market assumes."
            elif gap is not None and gap > 0:
                plain += (
                    f"It has historically grown at {trailing_cagr}% — "
                    f"the market is asking for {gap:.0f} percentage points more than it has delivered."
                )
            else:
                plain += (
                    f"It has historically grown at {trailing_cagr}% — "
                    f"the market is actually pricing in less growth than history suggests."
                )

        return json.dumps({
            "ticker": ticker,
            "name": info.get("longName"),
            "current_price": price,
            "market_cap": market_cap,
            "current_revenue": revenue,
            "current_fcf": round(fcf),
            "fcf_margin_pct": round(fcf_margin * 100, 1),
            "dcf_assumptions": {
                "discount_rate_pct": 10,
                "terminal_fcf_multiple": terminal_mult,
                "horizon_years": 10,
                "fcf_margin_held": "constant at current level",
            },
            "implied_10yr_revenue_cagr_pct": implied_cagr,
            "trailing_3yr_revenue_cagr_pct": trailing_cagr,
            "gap_pct": gap,
            "verdict": verdict,
            "plain_english": plain,
            "data_source": "Yahoo Finance via yfinance",
        })

    except Exception as e:
        return json.dumps({"error": f"DCF implied growth calculation failed: {e}"})


# ── 11. Dilution Rate ──────────────────────────────────────────────────────────
# Stock-based compensation (SBC) is a real cost to shareholders that doesn't
# appear in GAAP earnings. High-multiple IT stocks often look profitable until
# you subtract the equity being handed to employees. This tool makes that cost
# explicit: SBC as a % of revenue, SBC as a % of gross profit, the true FCF
# margin after SBC, and YoY share count growth. Retail investors who cite
# reported FCF without adjusting for SBC are looking at an inflated number.

dilution_tool = {
    "name": "get_dilution_rate",
    "description": (
        "Measure stock-based compensation (SBC) dilution: SBC as % of revenue, "
        "SBC as % of gross profit, SBC-adjusted FCF margin (FCF minus SBC / revenue), "
        "and YoY share count growth %. Benchmarks SBC level (low/moderate/high/very high) "
        "and flags if share count grew >3% YoY. Always use this alongside valuation tools "
        "— GAAP FCF overstates true cash generation when SBC is material."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'MSFT'"}
        },
        "required": ["ticker"]
    }
}

def get_dilution_rate(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        stock  = yf.Ticker(ticker)
        info   = stock.info
        cf     = stock.cashflow
        fin    = stock.financials

        result = {"ticker": ticker, "name": info.get("longName")}

        # ── SBC from cash flow statement ──────────────────────────────────────
        sbc = None
        for row_name in ["Stock Based Compensation",
                         "Share Based Compensation Expense",
                         "Stock Based Compensation Expense"]:
            if row_name in cf.index:
                s = cf.loc[row_name].dropna()
                if len(s) >= 1:
                    sbc = float(s.iloc[0])
                break

        if sbc is None:
            result["sbc_error"] = "SBC not found in cash flow statement for this ticker."

        # ── Revenue ───────────────────────────────────────────────────────────
        revenue = info.get("totalRevenue")
        if not revenue:
            try:
                revs = fin.loc["Total Revenue"].dropna()
                revenue = float(revs.iloc[0]) if len(revs) >= 1 else None
            except Exception:
                pass

        # ── Gross profit ──────────────────────────────────────────────────────
        gross_profit = None
        try:
            gp = fin.loc["Gross Profit"].dropna()
            gross_profit = float(gp.iloc[0]) if len(gp) >= 1 else None
        except Exception:
            pass

        # ── Free cash flow ────────────────────────────────────────────────────
        fcf = None
        try:
            f = cf.loc["Free Cash Flow"].dropna()
            fcf = float(f.iloc[0]) if len(f) >= 1 else None
        except Exception:
            pass

        # ── Share count: current vs prior year ───────────────────────────────
        shares_current = info.get("sharesOutstanding")
        shares_prior   = None
        shares_growth  = None

        for row_name in ["Diluted Average Shares", "Ordinary Shares Number",
                         "Share Issued", "Basic Average Shares"]:
            try:
                if row_name in fin.index:
                    s = fin.loc[row_name].dropna()
                    if len(s) >= 2:
                        shares_prior = float(s.iloc[1])
                        break
                if row_name in cf.index:
                    s = cf.loc[row_name].dropna()
                    if len(s) >= 2:
                        shares_prior = float(s.iloc[1])
                        break
            except Exception:
                continue

        if shares_current and shares_prior and shares_prior > 0:
            shares_growth = round((shares_current - shares_prior) / shares_prior * 100, 2)

        # ── Derived metrics ───────────────────────────────────────────────────
        sbc_pct_rev  = round(sbc / revenue * 100, 1)       if sbc and revenue       else None
        sbc_pct_gp   = round(sbc / gross_profit * 100, 1)  if sbc and gross_profit  else None
        gaap_fcf_mgn = round(fcf / revenue * 100, 1)       if fcf and revenue       else None
        sbc_adj_mgn  = round((fcf - sbc) / revenue * 100, 1) if fcf is not None and sbc and revenue else None
        overstate    = round(gaap_fcf_mgn - sbc_adj_mgn, 1) if gaap_fcf_mgn and sbc_adj_mgn else None

        # ── Benchmarks ────────────────────────────────────────────────────────
        if sbc_pct_rev is not None:
            if sbc_pct_rev < 5:
                sbc_bench = f"Low ({sbc_pct_rev}% of revenue) — minimal dilution drag on true profitability."
            elif sbc_pct_rev < 10:
                sbc_bench = f"Moderate ({sbc_pct_rev}% of revenue) — meaningful but manageable."
            elif sbc_pct_rev < 20:
                sbc_bench = f"High ({sbc_pct_rev}% of revenue) — significant drag; GAAP profits overstate shareholder returns."
            else:
                sbc_bench = f"Very High ({sbc_pct_rev}% of revenue) — major equity dilution. True profitability substantially lower than reported."
        else:
            sbc_bench = None

        if shares_growth is None:
            dilution_flag = "Share count history unavailable — could not calculate YoY dilution."
        elif shares_growth > 3:
            dilution_flag = (
                f"WARNING: Share count grew {shares_growth}% YoY — above the 3% warning threshold. "
                "Existing shareholders are being meaningfully diluted."
            )
        elif shares_growth > 0:
            dilution_flag = f"Share count grew {shares_growth}% YoY — modest dilution."
        else:
            dilution_flag = (
                f"Share count declined {abs(shares_growth)}% YoY — net buyback activity, accretive to shareholders."
            )

        result.update({
            "shares_outstanding_current": shares_current,
            "shares_outstanding_prior_year": round(shares_prior) if shares_prior else None,
            "share_count_growth_pct": shares_growth,
            "dilution_flag": dilution_flag,
            "sbc_annual": round(sbc) if sbc else None,
            "sbc_pct_revenue": sbc_pct_rev,
            "sbc_pct_gross_profit": sbc_pct_gp,
            "sbc_benchmark": sbc_bench,
            "gaap_fcf_margin_pct": gaap_fcf_mgn,
            "sbc_adjusted_fcf_margin_pct": sbc_adj_mgn,
            "gaap_fcf_overstatement_ppt": overstate,
            "data_source": "Yahoo Finance via yfinance",
        })
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Dilution rate calculation failed: {e}"})


# ══════════════════════════════════════════════════════════════════════════════
# AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    calculator_tool,
    stock_price_tool,
    company_info_tool,
    financial_data_tool,
    news_sentiment_tool,
    insider_trades_tool,
    earnings_surprise_tool,
    reddit_sentiment_tool,
    nrr_tool,
    dcf_implied_tool,
    dilution_tool,
]

ALL_FUNCTIONS = {
    "calculator": calculator,
    "get_stock_price": get_stock_price,
    "get_company_info": get_company_info,
    "get_financial_data": get_financial_data,
    "get_news_sentiment": get_news_sentiment,
    "get_insider_trades": get_insider_trades,
    "get_earnings_surprise": get_earnings_surprise,
    "get_reddit_sentiment": get_reddit_sentiment,
    "get_net_revenue_retention": get_net_revenue_retention,
    "get_dcf_implied_growth": get_dcf_implied_growth,
    "get_dilution_rate": get_dilution_rate,
}


def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    Run the stock research agent on a query.
    Loops until Claude stops requesting tool calls.
    Max 20 steps to prevent runaway loops.
    """
    if verbose:
        print(f"\n{'═' * 65}")
        print(f"  {user_message}")
        print(f"{'═' * 65}")

    messages = [{"role": "user", "content": user_message}]
    step = 0

    while True:
        step += 1

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=messages,
        )

        if verbose:
            print(f"\n── Step {step}  stop_reason={response.stop_reason}")

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "text" and block.text.strip() and verbose:
                    snippet = block.text[:200] + "..." if len(block.text) > 200 else block.text
                    print(f"   🧠 {snippet}")

                elif block.type == "tool_use":
                    if verbose:
                        print(f"   🛠  {block.name}({json.dumps(block.input)})")

                    fn = ALL_FUNCTIONS.get(block.name)
                    result = fn(**block.input) if fn else json.dumps({"error": f"Unknown tool: {block.name}"})

                    if verbose:
                        preview = result[:120] + "..." if len(result) > 120 else result
                        print(f"      → {preview}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            # Final answer — Claude is done calling tools
            answer = "".join(b.text for b in response.content if hasattr(b, "text"))
            print(f"\n{answer}")
            print(f"\n✅ Completed in {step} step(s)")
            return answer

        if step >= 20:
            err = "Error: agent exceeded 20 steps — possible loop. Aborting."
            print(err)
            return err


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run the three test queries
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("  STOCK RESEARCH AGENT — IT Sector Behavioral Finance Analysis")
    print("█" * 65)

    # ── Test 1: Full narrative vs reality report ───────────────────────────
    print("\n\n" + "▓" * 65)
    print("  TEST 1: Full Narrative vs Reality — TSLA")
    print("▓" * 65)
    run_agent("Give me a full narrative vs reality report on TSLA.")

    # ── Test 2: Institutional vs retail alignment comparison ───────────────
    print("\n\n" + "▓" * 65)
    print("  TEST 2: Institutional vs Retail Alignment — NVDA vs MSFT")
    print("▓" * 65)
    run_agent(
        "Compare NVDA and MSFT — which has stronger institutional vs retail alignment? "
        "Pull all relevant data and give me a side-by-side analysis."
    )

    # ── Test 3: Behavioral finance opportunity scan ────────────────────────
    print("\n\n" + "▓" * 65)
    print("  TEST 3: Behavioral Finance Opportunity — META")
    print("▓" * 65)
    run_agent(
        "Is there a behavioral finance opportunity in META right now based on "
        "sentiment vs fundamentals? Use all available tools."
    )

    # ── Test 4: DCF implied growth — NVDA valuation ───────────────────────
    print("\n\n" + "▓" * 65)
    print("  TEST 4: DCF Implied Growth — NVDA")
    print("▓" * 65)
    run_agent(
        "Is NVIDIA's current stock price pricing in growth that's historically achievable, "
        "or is it pricing in something exceptional? What has to be true for the valuation "
        "to be justified?"
    )

    # ── Test 5: SBC dilution — MSFT true FCF margin ───────────────────────
    print("\n\n" + "▓" * 65)
    print("  TEST 5: Dilution Rate — MSFT")
    print("▓" * 65)
    run_agent(
        "How much are Microsoft shareholders being diluted by stock-based compensation, "
        "and what is the true FCF margin after accounting for SBC?"
    )

    # ── Test 6: Net Revenue Retention — CRM (Salesforce) ─────────────────
    print("\n\n" + "▓" * 65)
    print("  TEST 6: Net Revenue Retention — CRM (Salesforce)")
    print("▓" * 65)
    run_agent(
        "For a SaaS company like Salesforce (CRM), estimate the net revenue retention "
        "and tell me whether their existing customer base is expanding or contracting."
    )
