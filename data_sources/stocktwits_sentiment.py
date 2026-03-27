#!/usr/bin/env python3
"""
stocktwits_sentiment.py — StockTwits sentiment via free public API.

No authentication required. Uses api.stocktwits.com/api/2/streams/symbol/{ticker}.json

Functions:
    get_stocktwits_sentiment(ticker) → dict
"""

import time
import requests
from datetime import datetime

from data_sources._cache import cache_get, cache_set, cache_key, log_fetch, logger


STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
HEADERS = {"User-Agent": "StockResearchAgent/1.0"}


def get_stocktwits_sentiment(ticker: str) -> dict:
    """
    Fetch recent StockTwits messages for a ticker and compute sentiment.
    Uses the messages[].entities.sentiment.basic field ("Bullish" / "Bearish").
    Unannotated messages are counted as neutral.
    """
    ck = cache_key("stocktwits", ticker)
    hit = cache_get(ck)
    if hit:
        log_fetch("StockTwits", ticker, cached=True)
        return hit

    t0 = time.time()

    try:
        url = STOCKTWITS_URL.format(ticker=ticker.upper())
        r = requests.get(url, headers=HEADERS, timeout=10)

        # 404 → ticker not found on StockTwits
        if r.status_code == 404:
            out = {
                "ticker":            ticker.upper(),
                "messages_analyzed": 0,
                "overall_sentiment": "neutral",
                "sentiment_score":   50,
                "signal_note":       "Ticker not found on StockTwits.",
                "data_source":       "StockTwits",
                "_elapsed_ms":       round((time.time() - t0) * 1000),
            }
            cache_set(ck, out)
            return out

        r.raise_for_status()
        data = r.json()

        messages   = data.get("messages", [])
        symbol_obj = data.get("symbol", {})
        watchers   = symbol_obj.get("watchlist_count", 0)

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        top_messages  = []

        for msg in messages:
            sentiment_obj = (msg.get("entities") or {}).get("sentiment") or {}
            basic = (sentiment_obj.get("basic") or "").strip()

            if basic == "Bullish":
                bullish_count += 1
            elif basic == "Bearish":
                bearish_count += 1
            else:
                neutral_count += 1

            # Collect top messages (non-empty body)
            body = (msg.get("body") or "").strip()
            if body and len(top_messages) < 5:
                top_messages.append({
                    "body":      body[:280],
                    "sentiment": basic or "neutral",
                    "likes":     msg.get("likes", {}).get("total", 0),
                    "created":   msg.get("created_at", "")[:10],
                    "username":  (msg.get("user") or {}).get("username", ""),
                })

        total = len(messages)
        annotated = bullish_count + bearish_count

        # Sentiment score: 0–100, 50 = neutral
        if annotated > 0:
            raw_score = bullish_count / annotated  # 0.0–1.0
            sentiment_score = round(raw_score * 100)
        else:
            sentiment_score = 50

        overall = ("bullish" if sentiment_score >= 60 else
                   "bearish" if sentiment_score <= 40 else "neutral")

        bull_bear_ratio = (round(bullish_count / bearish_count, 2)
                           if bearish_count > 0 else None)

        out = {
            "ticker":            ticker.upper(),
            "messages_analyzed": total,
            "bullish_count":     bullish_count,
            "bearish_count":     bearish_count,
            "neutral_count":     neutral_count,
            "bull_bear_ratio":   bull_bear_ratio,
            "overall_sentiment": overall,
            "sentiment_score":   sentiment_score,   # 0-100, 50=neutral
            "watchers":          watchers,
            "top_messages":      top_messages,
            "signal_note": (
                f"{bullish_count} bullish / {bearish_count} bearish out of "
                f"{total} recent messages. Watchlist: {watchers:,}."
                if total > 0 else "No recent StockTwits messages."
            ),
            "data_source":  "StockTwits (free public API)",
            "_elapsed_ms":  round((time.time() - t0) * 1000),
        }

        log_fetch("StockTwits", ticker, cached=False, elapsed_ms=out["_elapsed_ms"])
        cache_set(ck, out)
        return out

    except Exception as e:
        out = {"error": str(e), "ticker": ticker, "data_source": "StockTwits"}
        logger.error(f"get_stocktwits_sentiment({ticker}): {e}")
        return out
