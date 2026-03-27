#!/usr/bin/env python3
"""
tools_tech.py
-------------
IT sector-specific tools:
  - get_news_sentiment       — Yahoo Finance RSS headline scoring
  - get_earnings_surprise    — EPS beat/miss history
  - get_reddit_sentiment     — r/wallstreetbets + r/investing sentiment
  - get_net_revenue_retention — NRR proxy for SaaS/subscription businesses
"""

import re
import json
import time

import requests
import feedparser
import yfinance as yf
import pandas as pd

# ── News Sentiment ─────────────────────────────────────────────────────────────

news_sentiment_tool = {
    "name": "get_news_sentiment",
    "description": (
        "Fetch recent headlines from Yahoo Finance RSS and score them bullish/bearish/neutral "
        "via keyword analysis. Returns overall sentiment label, score -100 to +100, and top "
        "headlines. Use for IT, Healthcare, Consumer, and Energy sectors."
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
    "jumps", "climbs", "boosts", "wins", "exceeds", "better", "positive", "approved",
}
_NEWS_BEARISH = {
    "miss", "fall", "decline", "downgrade", "sell", "weak", "loss", "concern",
    "risk", "cut", "bearish", "underperform", "lowers", "drops", "plunges",
    "slides", "disappoints", "worse", "below", "warning", "layoffs", "lawsuit",
    "recall", "investigation", "probe", "rejected",
}

def get_news_sentiment(ticker: str) -> str:
    try:
        ticker = ticker.upper().strip()
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:15]:
            title   = entry.get("title", "")
            summary = entry.get("summary", "")[:200]
            words   = set(re.findall(r"\b\w+\b", (title + " " + summary).lower()))
            bull = words & _NEWS_BULLISH
            bear = words & _NEWS_BEARISH
            articles.append({
                "title": title,
                "published": entry.get("published", ""),
                "sentiment": "bullish" if len(bull) > len(bear) else "bearish" if len(bear) > len(bull) else "neutral",
                "signals": {"bullish": sorted(bull), "bearish": sorted(bear)},
            })
        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for a in articles:
            counts[a["sentiment"]] += 1
        total = len(articles) or 1
        score = round((counts["bullish"] - counts["bearish"]) / total * 100, 1)
        return json.dumps({
            "ticker": ticker,
            "articles_analyzed": len(articles),
            "overall_sentiment": "bullish" if score > 10 else "bearish" if score < -10 else "neutral",
            "sentiment_score": score,
            "breakdown": counts,
            "top_headlines": [
                {"title": a["title"], "sentiment": a["sentiment"], "published": a["published"]}
                for a in articles[:5]
            ],
            "data_source": "Yahoo Finance RSS",
        })
    except Exception as e:
        return json.dumps({"error": f"News fetch failed: {e}"})


# ── Earnings Surprise ──────────────────────────────────────────────────────────

earnings_surprise_tool = {
    "name": "get_earnings_surprise",
    "description": (
        "Last 4–6 quarters of EPS estimate vs actual — beat rate %, average surprise %, "
        "and per-quarter detail. Companies beating >75% of estimates command valuation premiums."
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
            return json.dumps({"error": f"No earnings data for '{ticker}'."})
        records = []
        for date, row in df.iterrows():
            actual_raw = row.get("Reported EPS")
            if pd.isna(actual_raw):
                continue
            try:
                estimate = float(row.get("EPS Estimate")) if not pd.isna(row.get("EPS Estimate")) else None
                actual   = float(actual_raw)
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
            return json.dumps({"error": f"No reported quarters for '{ticker}'."})
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


# ── Reddit Sentiment ───────────────────────────────────────────────────────────

reddit_sentiment_tool = {
    "name": "get_reddit_sentiment",
    "description": (
        "Search r/wallstreetbets and r/investing for posts mentioning a ticker in the last 7 days. "
        "Returns post volume, engagement, overall tone, and top posts. "
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
    all_posts = []
    for sub in ["wallstreetbets", "investing"]:
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
                })
            time.sleep(1)
        except Exception:
            pass
    if not all_posts:
        return json.dumps({
            "ticker": ticker, "posts_found": 0,
            "overall_sentiment": "neutral",
            "message": "No recent posts found — low retail attention signal.",
            "data_source": "Reddit public JSON API",
        })
    sent_scores = []
    for post in all_posts:
        words = set(re.findall(r"\b\w+\b", post["title"].lower()))
        bull = len(words & _WSB_BULLISH)
        bear = len(words & _WSB_BEARISH)
        sent_scores.append(1 if bull > bear else -1 if bear > bull else 0)
    avg = sum(sent_scores) / len(sent_scores)
    top = sorted(all_posts, key=lambda p: p["score"] + p["comments"] * 2, reverse=True)[:5]
    return json.dumps({
        "ticker": ticker,
        "posts_found": len(all_posts),
        "r_wallstreetbets": sum(1 for p in all_posts if p["subreddit"] == "wallstreetbets"),
        "r_investing": sum(1 for p in all_posts if p["subreddit"] == "investing"),
        "overall_sentiment": "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral",
        "sentiment_score": round(avg, 3),
        "top_posts": top,
        "data_source": "Reddit public JSON API",
    })


# ── Net Revenue Retention ──────────────────────────────────────────────────────
# Proxied via same-quarter YoY revenue growth and deferred revenue trends.
# Only meaningful for subscription/SaaS businesses.

nrr_tool = {
    "name": "get_net_revenue_retention",
    "description": (
        "Estimate NRR for SaaS/subscription businesses via same-quarter YoY revenue growth "
        "and deferred revenue trends. Returns estimate, benchmark tier, and trend direction. "
        "Returns a clear explanation if NRR is not applicable to the business model."
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
        stock  = yf.Ticker(ticker)
        info   = stock.info
        sector   = info.get("sector", "")
        industry = info.get("industry", "")
        summary  = (info.get("longBusinessSummary") or "").lower()

        sub_keywords = ["software", "saas", "cloud", "subscription", "platform",
                        "service", "recurring", "software-as"]
        is_subscription = (
            any(k in (sector + industry).lower() for k in sub_keywords)
            or any(k in summary for k in sub_keywords)
        )

        if not is_subscription:
            if "semiconductor" in industry.lower() or "hardware" in industry.lower():
                alt = "Design Win Rate and Customer Concentration"
            elif "retail" in industry.lower() or "consumer" in sector.lower():
                alt = "Same-Store Sales Growth (SSSG)"
            else:
                alt = "Customer Concentration and Revenue Cohort Analysis"
            return json.dumps({
                "ticker": ticker, "nrr_applicable": False,
                "sector": sector, "industry": industry,
                "explanation": (
                    f"{info.get('longName', ticker)} operates in {industry} ({sector}), "
                    "which is not a subscription-based model. NRR is not meaningful here."
                ),
                "recommended_metric_instead": alt,
                "data_source": "Yahoo Finance via yfinance",
            })

        results = {"ticker": ticker, "nrr_applicable": True, "sector": sector, "industry": industry}

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
                                "flat" if abs(dr_chg) <= 5 else "negative"
                            )
                    break
        except Exception:
            pass

        if "nrr_estimate_pct" not in results and "deferred_revenue_yoy_pct" not in results:
            return json.dumps({
                "ticker": ticker, "nrr_applicable": True,
                "error": "Insufficient historical data — need at least 5 quarters.",
                "data_source": "Yahoo Finance via yfinance",
            })

        nrr = results.get("nrr_estimate_pct")
        if nrr is not None:
            if nrr < 100:
                benchmark = "Below 100% — net churn; customer base is shrinking."
                health    = "poor"
            elif nrr < 110:
                benchmark = "100–110% — stable, marginal expansion."
                health    = "stable"
            elif nrr < 120:
                benchmark = "110–120% — healthy expansion."
                health    = "healthy"
            else:
                benchmark = "Above 120% — exceptional; existing customers drive organic growth."
                health    = "exceptional"
            results["benchmark"] = benchmark
            results["health_rating"] = health
            results["plain_english"] = (
                f"{info.get('longName', ticker)}'s existing customer base is "
                f"{'growing' if nrr >= 100 else 'shrinking'} at an estimated "
                f"{abs(nrr - 100):.1f}% net rate. "
                f"{'Above' if nrr >= 110 else 'Below'} the 110% benchmark for high-growth SaaS."
            )

        results["data_source"] = "Yahoo Finance via yfinance (proxy calculation)"
        return json.dumps(results)

    except Exception as e:
        return json.dumps({"error": f"NRR calculation failed: {e}"})


# ── Exports ────────────────────────────────────────────────────────────────────

TECH_TOOL_DEFS = [
    news_sentiment_tool,
    earnings_surprise_tool,
    reddit_sentiment_tool,
    nrr_tool,
]

TECH_FUNCTIONS = {
    "get_news_sentiment": get_news_sentiment,
    "get_earnings_surprise": get_earnings_surprise,
    "get_reddit_sentiment": get_reddit_sentiment,
    "get_net_revenue_retention": get_net_revenue_retention,
}
