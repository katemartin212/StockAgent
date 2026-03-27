"""
data_sources — modular free data-source integrations for the stock research agent.

Modules:
    _cache              Shared in-memory TTL cache + structured logging
    sec_edgar           SEC EDGAR XBRL financials, Form 4, recent filings
    fred_macro          FRED macro series (no API key required)
    open_insider        OpenInsider.com insider transaction scraper
    reddit_sentiment    Reddit sentiment (PRAW + public API fallback)
    stocktwits_sentiment StockTwits sentiment (free public API)
    trends_signal       Google Trends search interest (pytrends)
"""
