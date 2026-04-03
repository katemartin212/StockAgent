#!/usr/bin/env python3
"""
reddit_sentiment.py — High-signal Reddit sentiment with strict relevance filtering.

Eight-step protocol:
  1. Tiered subreddit prioritization (Tier 1 always; Tier 2 sector-specific)
  2. Relevance scoring 0-100, threshold 60
  3. Strict content extraction rules
  4. Claude API sentiment classification (batch, single API call)
  5. Weighted aggregate sentiment (relevance × upvotes)
  6. Structured fallback states for low-coverage tickers
  7. Rich display fields (flair, top_comment, relevance_score)
  8. 30-minute cache, 0.5s inter-subreddit delay, 429 backoff

Functions:
    get_reddit_sentiment(ticker, sector=None) → dict
"""

import os
import re
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from data_sources._cache import cache_get, cache_set, cache_key, log_fetch, logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Subreddit tiers ───────────────────────────────────────────────────────────

TIER1 = [
    "wallstreetbets",
    "investing",
    "stocks",
    "SecurityAnalysis",
    "options",
]

TIER2_BY_SECTOR = {
    "Technology":             ["technology", "hardware"],
    "Communication Services": ["technology"],
    "Healthcare":             ["biotech"],
    "Financial Services":     ["banking", "financialindependence"],
    "Real Estate":            ["RealEstate", "realestateinvesting"],
    "Energy":                 ["energy"],
}

# Posts from these subreddits are silently discarded regardless of content
_BLOCKLIST = {
    "fluentinfinance", "personalfinance", "financialindependence",
    "povertyfinance", "mildlyinfuriating", "news", "worldnews",
    "legaladvice", "askreddit", "politics", "worldpolitics",
}

# ── Relevance scoring ─────────────────────────────────────────────────────────

def _score_relevance(post: dict, ticker: str, company_name: str = "") -> int:
    """
    Score a post 0-100 for relevance to the ticker.
    Returns the score; posts below 60 are discarded.
    """
    title  = post.get("title", "")
    body   = post.get("body", "") or ""
    score  = post.get("score", 0) or 0
    comments = post.get("comments", 0) or 0

    t_upper = ticker.upper()
    title_upper = title.upper()
    body_upper  = body.upper()

    # Block non-tier subreddits immediately
    if post.get("subreddit", "").lower() in _BLOCKLIST:
        return 0

    # ── Ticker prominence (0-30) ──────────────────────────────────────────────
    ticker_points = 0
    dollar_pattern = rf"\${re.escape(t_upper)}\b"
    word_pattern   = rf"\b{re.escape(t_upper)}\b"

    if re.search(dollar_pattern, title, re.IGNORECASE):
        ticker_points = 30
    elif re.search(word_pattern, title_upper):
        # count other tickers in title to check if we're in a list
        other_tickers = len(re.findall(r"\$[A-Z]{1,5}\b", title)) - (1 if f"${t_upper}" in title.upper() else 0)
        ticker_points = 5 if other_tickers >= 5 else 20
    elif company_name and company_name.split()[0].lower() in title.lower():
        ticker_points = 10
    elif re.search(word_pattern, body_upper) or re.search(dollar_pattern, body, re.IGNORECASE):
        other_tickers = len(re.findall(r"\$[A-Z]{1,5}\b", title))
        ticker_points = 0 if other_tickers >= 5 else 5
    # else: 0

    # Early exit — if ticker barely appears, post is almost certainly irrelevant
    if ticker_points == 0:
        return 0

    # ── Post intent (0-30) ───────────────────────────────────────────────────
    intent_points = 0
    title_lower = title.lower()
    body_lower  = body.lower()

    # Count total distinct tickers in title+body
    all_tickers = set(re.findall(r"\$[A-Z]{1,5}\b", title + " " + body))
    n_tickers = len(all_tickers)

    # Signals of post being explicitly about this company
    explicit_signals = [
        any(kw in title_lower for kw in ["dd", "due diligence", "analysis", "thesis"]),
        any(kw in title_lower for kw in ["earnings", "q1", "q2", "q3", "q4", "guidance"]),
        any(kw in title_lower for kw in ["thoughts on", "opinion on", "what do you think", "should i buy", "should i sell"]),
        any(kw in title_lower for kw in ["price target", "pt ", "target price", "valuation"]),
        any(kw in title_lower for kw in ["buy", "sell", "hold", "short", "long"]) and n_tickers <= 3,
    ]

    if sum(explicit_signals) >= 1 and n_tickers <= 3:
        intent_points = 30
    elif n_tickers <= 3:
        intent_points = 20
    elif n_tickers <= 8:
        intent_points = 10
    else:
        intent_points = 0

    # ── Content quality (0-25) ───────────────────────────────────────────────
    word_count = len(body.split()) if body.strip() else 0

    # Discard content-free posts
    junk_phrases = {"no more please", "removed", "[deleted]", "[removed]"}
    if body.strip().lower() in junk_phrases or (not body.strip() and not title.strip()):
        return 0

    if word_count > 200:
        content_points = 25
    elif word_count >= 50:
        content_points = 15
    elif word_count >= 5 or not body.strip():
        content_points = 5
    else:
        content_points = 0

    # ── Engagement quality (0-15) ────────────────────────────────────────────
    if score >= 50 and comments >= 10:
        engagement_points = 15
    elif score >= 20 or comments >= 5:
        engagement_points = 10
    elif score >= 5:
        engagement_points = 5
    else:
        engagement_points = 0

    return ticker_points + intent_points + content_points + engagement_points


# ── Claude sentiment classification (batched) ─────────────────────────────────

def _claude_client():
    """Lazy Anthropic client — only instantiated when needed."""
    try:
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    except Exception:
        return None


def _classify_batch(posts: list[dict], ticker: str) -> list[str]:
    """
    Classify sentiment for a batch of posts with a single Claude API call.
    Returns list of 'bullish'|'bearish'|'neutral' aligned to posts.
    Falls back to keyword heuristic if API call fails.
    """
    if not posts:
        return []

    client = _claude_client()
    if client is None:
        return [_keyword_classify(p["title"] + " " + p.get("body", "")) for p in posts]

    numbered = []
    for i, p in enumerate(posts, 1):
        snippet = f"Title: {p['title']}\nBody: {(p.get('body') or '')[:500]}"
        numbered.append(f"Post {i}:\n{snippet}")

    prompt = (
        f"You are a financial sentiment classifier. Classify the sentiment of each Reddit post "
        f"about {ticker.upper()} as exactly one of: bullish, bearish, or neutral.\n"
        f"Bullish = author believes stock will go up or company has positive prospects.\n"
        f"Bearish = author believes stock will go down or company has negative prospects.\n"
        f"Neutral = author is asking a question, sharing news without opinion, or mixed views.\n\n"
        f"Respond with ONLY a JSON array of strings, one per post in order, e.g. "
        f'["bullish","neutral","bearish"]. No explanation.\n\n'
        + "\n\n---\n\n".join(numbered)
    )

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",   # fastest/cheapest for classification
            max_tokens=64 + len(posts) * 12,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Parse JSON array
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            labels = json.loads(m.group())
            # Normalize and pad/trim to match post count
            valid = {"bullish", "bearish", "neutral"}
            result = []
            for lbl in labels:
                result.append(lbl.lower() if lbl.lower() in valid else "neutral")
            while len(result) < len(posts):
                result.append("neutral")
            return result[:len(posts)]
    except Exception as e:
        logger.warning(f"Claude batch classify failed: {e}")

    # Keyword fallback
    return [_keyword_classify(p["title"] + " " + p.get("body", "")) for p in posts]


_BULLISH_WORDS = {
    "buy", "long", "calls", "bull", "bullish", "moon", "rocket",
    "undervalued", "accumulate", "upside", "strong buy", "dip",
}
_BEARISH_WORDS = {
    "sell", "short", "puts", "bear", "bearish", "overvalued", "dump",
    "crash", "downside", "avoid",
}

def _keyword_classify(text: str) -> str:
    t = text.lower()
    b = sum(1 for w in _BULLISH_WORDS if w in t)
    s = sum(1 for w in _BEARISH_WORDS if w in t)
    if b > s: return "bullish"
    if s > b: return "bearish"
    return "neutral"


# ── Reddit fetching ───────────────────────────────────────────────────────────

def _fetch_subreddit(sub: str, query: str, headers: dict) -> list[dict]:
    """
    Fetch up to 25 posts from one subreddit with 429 backoff.
    Returns list of raw post dicts (may be empty on error).
    """
    url = f"https://www.reddit.com/r/{sub}/search.json?q={query}&sort=relevance&t=month&limit=25&restrict_sr=1"
    for attempt in range(2):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 429:
                logger.warning(f"Reddit 429 on r/{sub} — backing off 60s")
                time.sleep(60)
                continue
            if r.status_code != 200:
                logger.warning(f"Reddit r/{sub} returned {r.status_code}")
                return []
            data = r.json()
            posts = []
            for child in data.get("data", {}).get("children", []):
                p = child.get("data", {})
                if not p.get("title"):
                    continue
                posts.append({
                    "title":     p.get("title", ""),
                    "subreddit": p.get("subreddit", sub),
                    "score":     p.get("score", 0),
                    "comments":  p.get("num_comments", 0),
                    "body":      (p.get("selftext", "") or "")[:800],
                    "flair":     p.get("link_flair_text") or None,
                    "url":       p.get("url", ""),
                    "created":   datetime.fromtimestamp(
                        p.get("created_utc", 0)
                    ).strftime("%Y-%m-%d"),
                    "_post_id":  p.get("id", ""),
                })
            return posts
        except Exception as e:
            logger.warning(f"Reddit fetch error r/{sub}: {e}")
            return []
    return []


def _fetch_top_comment(post_id: str, sub: str, headers: dict) -> str | None:
    """Fetch the top comment on a post. Returns None on failure."""
    if not post_id:
        return None
    try:
        url = f"https://www.reddit.com/r/{sub}/comments/{post_id}.json?limit=5&sort=best"
        time.sleep(0.3)
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        # data[1] = comments listing
        if len(data) < 2:
            return None
        for child in data[1].get("data", {}).get("children", []):
            body = child.get("data", {}).get("body", "")
            if body and body not in ("[deleted]", "[removed]") and len(body.split()) >= 10:
                return body[:200]
    except Exception:
        pass
    return None


# ── Fallback state detection ──────────────────────────────────────────────────

# Known large-cap tickers that tend to have low WSB retail presence
_INSTITUTIONAL_NAMES = {
    "BRK-B", "BRK-A", "BRK", "BRK.B", "BRK.A",
    "JNJ", "PG", "KO", "WMT", "UNH", "XOM",
    "V", "MA", "JPM", "BAC", "WFC",
}

def _fallback_state(ticker: str, total_fetched: int) -> dict:
    """Determine which fallback state applies when < 3 relevant posts found."""
    t = ticker.upper().replace(".", "-")
    if t in _INSTITUTIONAL_NAMES or total_fetched > 25:
        state   = "low_retail_coverage"
        note    = "Low retail coverage — institutional name. Low retail interest in a quality company is often a contrarian buy indicator."
        signal  = "accumulation_candidate"
    elif total_fetched >= 5:
        state   = "insufficient_discussion"
        note    = "Insufficient relevant discussion found in monitored communities. Posts mention ticker incidentally rather than as primary subject."
        signal  = "neutral"
    elif total_fetched > 0:
        state   = "emerging_name"
        note    = "Emerging name — building retail awareness. Limited social media history suggests early-stage institutional coverage."
        signal  = "neutral"
    else:
        state   = "no_coverage"
        note    = "No relevant posts found in monitored communities."
        signal  = "neutral"
    return {"state": state, "signal_note": note, "behavioral_signal": signal}


# ── Main entry point ──────────────────────────────────────────────────────────

def get_reddit_sentiment(ticker: str, sector: str | None = None, company_name: str = "") -> dict:
    """
    Fetch and filter Reddit posts for a ticker using the 8-step protocol.
    Cache TTL: 30 minutes.
    """
    ticker = ticker.upper()

    ck  = cache_key("reddit_v2", ticker)
    hit = cache_get(ck, ttl=1800)
    if hit:
        log_fetch("Reddit", ticker, cached=True)
        return hit

    t0 = time.time()
    headers = {"User-Agent": "StockResearchAgent/1.0 (research tool, not commercial)"}

    # ── Step 1: Build subreddit list ─────────────────────────────────────────
    tier1_subs = list(TIER1)
    tier2_subs = TIER2_BY_SECTOR.get(sector or "", [])

    # ── Fetch from all subreddits in parallel ────────────────────────────────
    # Use broader query to capture both $TICKER and plain TICKER
    query = f"{ticker} OR ${ticker}"

    all_raw: list[dict] = []
    fetched_from: dict[str, int] = {}

    def _fetch_with_tier(sub: str, tier: int) -> tuple[str, int, list[dict]]:
        posts = _fetch_subreddit(sub, query, headers)
        for p in posts:
            p["_tier"] = tier
        return sub, tier, posts

    all_subs = [(s, 1) for s in tier1_subs] + [(s, 2) for s in tier2_subs]
    with ThreadPoolExecutor(max_workers=len(all_subs)) as _pool:
        _futs = {_pool.submit(_fetch_with_tier, sub, tier): sub for sub, tier in all_subs}
        for _fut in as_completed(_futs, timeout=12):
            try:
                sub, tier, posts = _fut.result()
                fetched_from[sub] = len(posts)
                all_raw.extend(posts)
            except Exception as e:
                logger.warning(f"Reddit subreddit fetch failed: {e}")

    total_fetched = len(all_raw)

    # ── Step 2: Relevance scoring ─────────────────────────────────────────────
    for p in all_raw:
        p["relevance_score"] = _score_relevance(p, ticker, company_name)

    # Filter: must score >= 60; discard blocklisted subreddits
    relevant = [p for p in all_raw if p["relevance_score"] >= 60]

    # Deduplicate by post ID
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for p in relevant:
        pid = p.get("_post_id", p["title"][:40])
        if pid not in seen_ids:
            seen_ids.add(pid)
            deduped.append(p)

    # ── Fallback if < 3 pass ──────────────────────────────────────────────────
    low_coverage = len(deduped) < 3
    fallback = _fallback_state(ticker, total_fetched) if low_coverage else {}

    # ── Step 4: Claude API sentiment classification ───────────────────────────
    # Sort by relevance × engagement before classifying top posts
    deduped.sort(
        key=lambda p: (p["relevance_score"] * 0.6) + (min(p["score"], 500) * 0.4),
        reverse=True,
    )
    top_n = deduped[:10]   # classify up to 10 posts, show top 5

    sentiments = _classify_batch(top_n, ticker)
    for p, sent in zip(top_n, sentiments):
        p["sentiment"] = sent

    # ── Step 3+7: Top comment fetch for top 3 posts ───────────────────────────
    for p in top_n[:3]:
        pid = p.get("_post_id")
        sub = p.get("subreddit", "")
        if pid:
            p["top_comment"] = _fetch_top_comment(pid, sub, headers)
        else:
            p["top_comment"] = None

    # ── Step 5: Weighted sentiment aggregation ────────────────────────────────
    bullish_w = bearish_w = neutral_w = 0.0
    total_weight = 0.0
    for p in top_n:
        # Weight = relevance_score × log(upvotes+1) × tier multiplier
        import math
        upvote_factor = math.log1p(max(p.get("score", 0), 0))
        tier_mult = 2.0 if p.get("_tier") == 1 else 1.0
        w = p["relevance_score"] * upvote_factor * tier_mult
        sent = p.get("sentiment", "neutral")
        if sent == "bullish":   bullish_w += w
        elif sent == "bearish": bearish_w += w
        else:                   neutral_w += w
        total_weight += w

    if total_weight > 0:
        bull_frac = bullish_w / total_weight
        bear_frac = bearish_w / total_weight
        weighted_score = round((bull_frac - bear_frac + 1) / 2 * 100)  # 0-100
        overall = ("bullish" if bull_frac > bear_frac + 0.1 else
                   "bearish" if bear_frac > bull_frac + 0.1 else "neutral")
    else:
        weighted_score = 50
        overall = "neutral"

    # Sentiment breakdown counts (from classified posts only)
    bull_count = sum(1 for p in top_n if p.get("sentiment") == "bullish")
    bear_count = sum(1 for p in top_n if p.get("sentiment") == "bearish")
    neut_count = len(top_n) - bull_count - bear_count

    subreddit_breakdown = {}
    for p in deduped:
        sub = p.get("subreddit", "unknown")
        subreddit_breakdown[sub] = subreddit_breakdown.get(sub, 0) + 1

    # Display-ready top posts
    display_posts = []
    for p in top_n[:5]:
        display_posts.append({
            "title":           p["title"],
            "subreddit":       p["subreddit"],
            "score":           p["score"],
            "comments":        p["comments"],
            "sentiment":       p.get("sentiment", "neutral"),
            "relevance_score": p["relevance_score"],
            "flair":           p.get("flair"),
            "top_comment":     p.get("top_comment"),
            "created":         p.get("created"),
            "body":            (p.get("body") or "")[:300],
        })

    # ── Step 6: signal_note ───────────────────────────────────────────────────
    if low_coverage:
        signal_note = fallback.get("signal_note", "Insufficient relevant posts.")
        behavioral_signal = fallback.get("behavioral_signal", "neutral")
    else:
        signal_note = (
            f"Showing {len(top_n)} of {total_fetched} posts fetched · "
            f"weighted sentiment: {weighted_score}/100 · "
            f"{bull_count} bullish, {bear_count} bearish, {neut_count} neutral"
        )
        behavioral_signal = overall

    elapsed = round((time.time() - t0) * 1000)
    out = {
        "ticker":               ticker,
        "posts_found":          len(top_n),
        "total_fetched":        total_fetched,
        "posts_passed_filter":  len(deduped),
        "low_coverage":         low_coverage,
        "fallback_state":       fallback.get("state") if low_coverage else None,
        "overall_sentiment":    overall,
        "sentiment_score":      round((bull_frac - bear_frac) if total_weight > 0 else 0, 3),
        "weighted_score":       weighted_score,
        "sentiment_breakdown": {
            "bullish_count": bull_count,
            "bearish_count": bear_count,
            "neutral_count": neut_count,
        },
        "subreddit_breakdown":  subreddit_breakdown,
        "engagement_score":     sum(p["score"] + p["comments"] * 2 for p in top_n),
        "top_posts":            display_posts,
        "auth_method":          "public_api",
        "signal_note":          signal_note,
        "behavioral_signal":    behavioral_signal,
        "data_source":          "Reddit (public API, strict relevance filter)",
        "_elapsed_ms":          elapsed,
    }
    log_fetch("Reddit", ticker, cached=False, elapsed_ms=elapsed)
    cache_set(ck, out)
    return out
