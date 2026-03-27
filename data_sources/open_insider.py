#!/usr/bin/env python3
"""
open_insider.py — Scrape OpenInsider.com for insider transaction data.

Complements SEC EDGAR Form 4 data with cluster-buy detection and a
pre-formatted transaction table.

Functions:
    get_open_insider(ticker) → dict
"""

import time
import re
import requests
from datetime import datetime, timedelta

from data_sources._cache import cache_get, cache_set, cache_key, log_fetch, logger

try:
    from bs4 import BeautifulSoup
    _BS_AVAILABLE = True
except ImportError:
    _BS_AVAILABLE = False

OPENINSIDER_URL = (
    "http://openinsider.com/screener"
    "?s={ticker}"
    "&fd=90&fdr=&td=0&tdr="
    "&xp=1&xs=1"          # include purchases and sales
    "&sortcol=0&cnt=40&action=1"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StockResearchAgent/1.0; contact@example.com)",
    "Accept": "text/html,application/xhtml+xml",
}


def get_open_insider(ticker: str) -> dict:
    """
    Scrape OpenInsider for the last 90 days of insider transactions.
    Detects cluster-buy events (3+ insiders buying within 30 days).
    """
    if not _BS_AVAILABLE:
        return {
            "error": "beautifulsoup4 not installed. Run: pip install beautifulsoup4 lxml",
            "ticker": ticker,
            "data_source": "OpenInsider.com",
        }

    ck = cache_key("open_insider", ticker)
    hit = cache_get(ck)
    if hit:
        log_fetch("OpenInsider", ticker, cached=True)
        return hit

    t0 = time.time()

    try:
        url = OPENINSIDER_URL.format(ticker=ticker.upper())
        time.sleep(0.5)   # polite delay for scraping
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "lxml")

        # OpenInsider uses a table with class "tinytable"
        table = soup.find("table", class_="tinytable")
        if not table:
            out = {
                "ticker":       ticker.upper(),
                "transactions_found": 0,
                "transactions": [],
                "signal":       "no_data",
                "signal_note":  "No insider transaction table found on OpenInsider.",
                "data_source":  "OpenInsider.com (scraped)",
                "_elapsed_ms":  round((time.time() - t0) * 1000),
            }
            cache_set(ck, out)
            return out

        rows = table.find_all("tr")[1:]  # skip header

        transactions = []
        net_value_purchased = 0
        net_value_sold      = 0

        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) < 12:
                continue

            try:
                # Column order on OpenInsider:
                # 0: X (checkbox), 1: Filing Date, 2: Trade Date, 3: Ticker,
                # 4: Company, 5: Insider Name, 6: Title, 7: Trade Type,
                # 8: Price, 9: Qty, 10: Owned, 11: ΔOwn, 12: Value
                filing_date = cols[1]
                trade_date  = cols[2]
                name        = cols[5]
                title       = cols[6]
                trade_type  = cols[7]  # e.g. "S - Sale", "P - Purchase"
                price_str   = cols[8].replace("$", "").replace(",", "")
                qty_str     = cols[9].replace(",", "").replace("+", "").replace("-", "")
                owned_str   = cols[10].replace(",", "")
                value_str   = cols[12].replace("$", "").replace(",", "").replace("+", "").replace("-", "")

                price    = float(price_str)  if price_str  else 0.0
                qty      = int(qty_str)      if qty_str    else 0
                owned    = int(owned_str)    if owned_str  else 0
                value    = int(float(value_str)) if value_str else abs(int(qty * price))

                is_purchase = trade_type.upper().startswith("P")
                is_sale     = trade_type.upper().startswith("S")

                if is_purchase:
                    net_value_purchased += value
                elif is_sale:
                    net_value_sold += value

                transactions.append({
                    "filing_date":      filing_date,
                    "trade_date":       trade_date,
                    "insider_name":     name,
                    "title":            title,
                    "transaction_type": trade_type,
                    "shares":           qty,
                    "price":            price,
                    "value_usd":        value,
                    "owns_after":       owned,
                    "is_purchase":      is_purchase,
                })
            except Exception:
                continue

        # Cluster buy detection: 3+ distinct insiders buying within a 30-day window
        buy_txns = [t for t in transactions if t["is_purchase"]]
        cluster_buy = False
        if len(buy_txns) >= 3:
            # Group by trade_date
            try:
                buy_events = []
                for t in buy_txns:
                    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
                        try:
                            buy_events.append((datetime.strptime(t["trade_date"], fmt), t["insider_name"]))
                            break
                        except ValueError:
                            continue
                buy_events.sort(key=lambda x: x[0])
                for i in range(len(buy_events) - 2):
                    d0, n0 = buy_events[i]
                    d2, n2 = buy_events[i+2]
                    names_in_window = {buy_events[j][1] for j in range(i, i+3)}
                    if (d2 - d0).days <= 30 and len(names_in_window) >= 3:
                        cluster_buy = True
                        break
            except Exception:
                pass

        net_flow = net_value_purchased - net_value_sold

        # Signal
        if not transactions:
            signal, note = "no_data", "No transactions found on OpenInsider."
        elif cluster_buy:
            signal = "cluster_buy"
            note = (f"CLUSTER BUY: 3+ insiders purchased within 30 days — "
                    f"highest-conviction signal. Net buy: ${net_value_purchased:,.0f}.")
        elif net_flow > 500_000:
            signal = "net_buying"
            note = f"Net insider buying of ${net_flow:,.0f} over 90 days."
        elif net_flow < -500_000:
            signal = "net_selling"
            note = f"Net insider selling of ${abs(net_flow):,.0f} over 90 days."
        else:
            signal = "neutral"
            note = "Minimal net insider activity."

        out = {
            "ticker":                  ticker.upper(),
            "transactions_found":      len(transactions),
            "net_value_purchased_usd": net_value_purchased,
            "net_value_sold_usd":      -net_value_sold,
            "net_flow_usd":            net_flow,
            "cluster_buy_signal":      cluster_buy,
            "transactions":            transactions[:15],
            "signal":                  signal,
            "signal_note":             note,
            "data_source":             "OpenInsider.com (scraped)",
            "_elapsed_ms":             round((time.time() - t0) * 1000),
        }
        log_fetch("OpenInsider", ticker, cached=False, elapsed_ms=out["_elapsed_ms"])
        cache_set(ck, out)
        return out

    except Exception as e:
        out = {"error": str(e), "ticker": ticker, "data_source": "OpenInsider.com"}
        logger.error(f"get_open_insider({ticker}): {e}")
        return out
