#!/usr/bin/env python3
"""
sec_edgar.py — SEC EDGAR data: XBRL financials, Form 4 insider transactions, recent filings.

All endpoints are free and require no API key.
EDGAR requires a descriptive User-Agent header per their policy.

Functions:
    get_edgar_financials(ticker)  — 8 quarters of XBRL financial time-series
    get_edgar_form4(ticker)       — Form 4 insider transaction detail (last 90 days)
    get_edgar_filings(ticker)     — Most recent filings (any form type)
"""

import re
import time
import json
import requests
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

from data_sources._cache import cache_get, cache_set, cache_key, log_fetch, logger

EDGAR_HEADERS = {
    "User-Agent": "StockResearchAgent/1.0 contact@stockresearch.example.com",
    "Accept-Encoding": "gzip, deflate",
}
REQUEST_DELAY = 0.12   # SEC allows ≤10 req/sec

TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL  = "https://data.sec.gov/submissions/CIK{cik}.json"
XBRL_URL         = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVES_URL     = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"

# XBRL concept candidates (try in order until one has data)
_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
_GROSS_PROFIT_TAGS = ["GrossProfit"]
_OP_INCOME_TAGS    = ["OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"]
_NET_INCOME_TAGS   = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
_OCF_TAGS          = ["NetCashProvidedByUsedInOperatingActivities"]
_CAPEX_TAGS        = ["PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpenditureDiscontinuedOperations"]
_SHARES_TAGS       = ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"]

# Form 4 transaction codes
_TX_WEIGHTS = {
    "P": +3.0,   # open-market purchase — highest conviction
    "S": -1.0,   # sale
    "M": +0.3,   # option exercise
    "G":  0.0,   # gift
    "F":  0.0,   # tax withholding
    "D":  0.0,   # disposition to company
    "A": +0.1,   # award (grant)
    "J":  0.0,   # other
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 15) -> requests.Response:
    time.sleep(REQUEST_DELAY)
    r = requests.get(url, headers=EDGAR_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


def _get_cik(ticker: str) -> str | None:
    """Zero-padded 10-digit CIK string, or None."""
    ck = cache_key("edgar_cik", ticker)
    hit = cache_get(ck, ttl=86400)   # CIK doesn't change — 24h cache
    if hit:
        return hit.get("cik")
    try:
        r = _get(TICKERS_JSON_URL)
        data = r.json()
        tu = ticker.upper()
        for _, company in data.items():
            if company.get("ticker", "").upper() == tu:
                cik = str(company["cik_str"]).zfill(10)
                cache_set(ck, {"cik": cik})
                return cik
    except Exception as e:
        logger.warning(f"CIK lookup failed for {ticker}: {e}")
    return None


def _get_submissions(cik: str) -> dict:
    """Fetch the EDGAR submissions JSON for a CIK."""
    url = SUBMISSIONS_URL.format(cik=cik)
    r = _get(url)
    return r.json()


def _pick_tag(facts_usgaap: dict, candidates: list[str]) -> list[dict] | None:
    """Return the unit array for the first candidate tag that has data."""
    for tag in candidates:
        node = facts_usgaap.get(tag)
        if node:
            units = node.get("units", {})
            usd   = units.get("USD") or units.get("shares")
            if usd:
                return usd
    return None


def _quarterly_series(raw: list[dict], n: int = 8) -> list[dict]:
    """
    Extract up to `n` most-recent quarterly / annual filing periods.
    Filters to 10-Q and 10-K forms, deduplicates by end date (keeps most recent).
    """
    relevant = [e for e in raw if e.get("form") in ("10-Q", "10-K")]
    by_end: dict[str, dict] = {}
    for e in relevant:
        end = e.get("end", "")
        existing = by_end.get(end)
        if not existing or e.get("filed", "") > existing.get("filed", ""):
            by_end[end] = e
    sorted_entries = sorted(by_end.values(), key=lambda x: x.get("end", ""), reverse=True)
    return sorted_entries[:n]


# ── Public functions ──────────────────────────────────────────────────────────

def get_edgar_financials(ticker: str) -> dict:
    """
    Pull 8 quarters of XBRL financial data from SEC EDGAR.
    Returns time-series for revenue, operating income, net income, OCF, capex, shares.
    """
    ck = cache_key("edgar_xbrl", ticker)
    hit = cache_get(ck)
    if hit:
        log_fetch("SEC EDGAR XBRL", ticker, cached=True)
        return hit

    t0 = time.time()

    try:
        cik = _get_cik(ticker)
        if not cik:
            out = {"error": f"CIK not found for {ticker}", "ticker": ticker,
                   "data_source": "SEC EDGAR XBRL"}
            return out

        url = XBRL_URL.format(cik=cik)
        r = _get(url, timeout=25)
        xbrl = r.json()
        facts = xbrl.get("facts", {})
        usgaap = facts.get("us-gaap", {})
        dei    = facts.get("dei", {})

        def extract(candidates, tag_pool=usgaap):
            raw = _pick_tag(tag_pool, candidates)
            if raw is None:
                return []
            return _quarterly_series(raw)

        # Build time-series for each metric
        rev_series     = extract(_REVENUE_TAGS)
        gp_series      = extract(_GROSS_PROFIT_TAGS)
        oi_series      = extract(_OP_INCOME_TAGS)
        ni_series      = extract(_NET_INCOME_TAGS)
        ocf_series     = extract(_OCF_TAGS)
        capex_series   = extract(_CAPEX_TAGS)
        shares_series  = extract(_SHARES_TAGS, {**usgaap, **dei})

        def to_clean(series, scale=1_000_000):
            """Convert raw XBRL entries to clean dicts with values in millions."""
            out = []
            for e in series:
                val = e.get("val")
                if val is None:
                    continue
                out.append({
                    "end":  e.get("end", ""),
                    "form": e.get("form", ""),
                    "val":  round(val / scale, 2),
                })
            return out

        revenue   = to_clean(rev_series)
        gp        = to_clean(gp_series)
        op_income = to_clean(oi_series)
        net_inc   = to_clean(ni_series)
        ocf       = to_clean(ocf_series)
        capex     = to_clean(capex_series)
        shares    = to_clean(shares_series, scale=1_000_000)  # in millions of shares

        # Compute FCF = OCF - capex for matching periods
        ocf_map   = {e["end"]: e["val"] for e in ocf}
        capex_map = {e["end"]: abs(e["val"]) for e in capex}
        fcf = [
            {"end": end, "val": round(v - capex_map.get(end, 0), 2)}
            for end, v in ocf_map.items()
        ]
        fcf.sort(key=lambda x: x["end"], reverse=True)

        # Revenue trend (most recent vs prior year annual)
        rev_trend = "insufficient_data"
        if len(revenue) >= 5:
            curr  = revenue[0]["val"]
            prior = revenue[4]["val"]
            if prior > 0:
                yoy = round((curr - prior) / prior * 100, 1)
                rev_trend = (f"+{yoy}%" if yoy >= 0 else f"{yoy}%")

        entity_name = xbrl.get("entityName", "")

        out = {
            "ticker":          ticker.upper(),
            "cik":             cik,
            "entity_name":     entity_name,
            "periods_available": len(revenue),
            "revenue_qtr_M":  revenue[:8],
            "gross_profit_M": gp[:8],
            "op_income_M":    op_income[:8],
            "net_income_M":   net_inc[:8],
            "ocf_M":          ocf[:8],
            "capex_M":        capex[:8],
            "fcf_M":          fcf[:8],
            "shares_M":       shares[:4],
            "revenue_yoy_trend": rev_trend,
            "data_source":    "SEC EDGAR XBRL API (data.sec.gov)",
            "_elapsed_ms":    round((time.time() - t0) * 1000),
        }
        log_fetch("SEC EDGAR XBRL", ticker, cached=False, elapsed_ms=out["_elapsed_ms"])
        cache_set(ck, out)
        return out

    except Exception as e:
        out = {"error": str(e), "ticker": ticker, "data_source": "SEC EDGAR XBRL"}
        logger.error(f"get_edgar_financials({ticker}): {e}")
        return out


def get_edgar_form4(ticker: str) -> dict:
    """
    Pull Form 4 insider transactions from the last 90 days.
    Classifies transactions by type and computes net dollar flow.
    Weighted signal: open-market purchases (P) weighted 3× more than exercises (M).
    """
    ck = cache_key("edgar_form4", ticker)
    hit = cache_get(ck)
    if hit:
        log_fetch("SEC EDGAR Form4", ticker, cached=True)
        return hit

    t0 = time.time()

    try:
        cik = _get_cik(ticker)
        if not cik:
            return {"error": f"CIK not found for {ticker}", "ticker": ticker,
                    "data_source": "SEC EDGAR Form 4"}

        subs   = _get_submissions(cik)
        recent = subs.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs  = recent.get("accessionNumber", [])

        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        form4_filings = [
            (d, a) for f, d, a in zip(forms, dates, accs)
            if f == "4" and d >= cutoff
        ]

        transactions = []
        net_weight   = 0.0
        net_flow_usd = 0.0

        for filing_date, accn in form4_filings[:20]:  # cap at 20 for speed
            try:
                accn_nd = accn.replace("-", "")
                # Try the standard Form 4 XML filename patterns
                xml_url = None
                for suffix in [".xml", "-index.htm"]:
                    candidate = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nd}/{accn}{suffix}"
                    # Get filing index to find actual XML
                try:
                    idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nd}/"
                    time.sleep(REQUEST_DELAY)
                    idx_r = requests.get(idx_url, headers=EDGAR_HEADERS, timeout=10)
                    # Find .xml link in index
                    xml_links = re.findall(r'href="([^"]+\.xml)"', idx_r.text, re.I)
                    if not xml_links:
                        continue
                    xml_url = "https://www.sec.gov" + xml_links[0] if xml_links[0].startswith("/") else xml_links[0]
                except Exception:
                    continue

                time.sleep(REQUEST_DELAY)
                xml_r = requests.get(xml_url, headers=EDGAR_HEADERS, timeout=10)
                root  = ET.fromstring(xml_r.text)

                # Extract filer info
                ns_map = {"": ""}
                rpt_owner = root.find(".//reportingOwner")
                filer_name  = ""
                filer_title = ""
                if rpt_owner is not None:
                    name_el  = rpt_owner.find(".//rptOwnerName")
                    title_el = rpt_owner.find(".//officerTitle")
                    if name_el  is not None: filer_name  = (name_el.text  or "").strip()
                    if title_el is not None: filer_title = (title_el.text or "").strip()

                # Parse non-derivative transactions
                for tx in root.findall(".//nonDerivativeTransaction"):
                    try:
                        code_el  = tx.find(".//transactionCode")
                        shares_el= tx.find(".//transactionShares/value")
                        price_el = tx.find(".//transactionPricePerShare/value")
                        date_el  = tx.find(".//transactionDate/value")

                        code   = (code_el.text  or "").strip().upper() if code_el  is not None else ""
                        shares = float(shares_el.text) if shares_el is not None and shares_el.text else 0
                        price  = float(price_el.text)  if price_el  is not None and price_el.text  else 0
                        date   = (date_el.text  or filing_date).strip() if date_el is not None else filing_date

                        dollar_val   = round(shares * price)
                        tx_weight    = _TX_WEIGHTS.get(code, 0.0)
                        net_weight  += tx_weight
                        if code == "P":
                            net_flow_usd += dollar_val
                        elif code == "S":
                            net_flow_usd -= dollar_val

                        if code in ("P", "S", "M"):
                            transactions.append({
                                "date":           date,
                                "filer":          filer_name,
                                "title":          filer_title,
                                "transaction_type": code,
                                "tx_label":       {"P": "Open-Market Purchase", "S": "Sale", "M": "Option Exercise"}.get(code, code),
                                "shares":         int(shares),
                                "price_per_share": price,
                                "dollar_value":   dollar_val,
                            })
                    except Exception:
                        continue

            except Exception as e:
                logger.debug(f"Form 4 parse error for {accn}: {e}")
                continue

        # Signal classification
        purchases = sum(1 for t in transactions if t["transaction_type"] == "P")
        sales      = sum(1 for t in transactions if t["transaction_type"] == "S")

        if not transactions:
            signal, note = "neutral", "No classified transactions found in the last 90 days."
        elif net_weight > 3:
            signal, note = "buying", f"Net insider buying. {purchases} open-market purchase(s) — conviction signal."
        elif net_weight < -1:
            signal, note = "selling", f"Net insider selling. {sales} sale(s) in last 90 days."
        elif purchases > 0 and sales > 0:
            signal, note = "mixed", f"{purchases} purchase(s) and {sales} sale(s) — mixed signals."
        elif purchases == 0 and sales == 0:
            signal, note = "awards_only", "Only option exercises or awards; no open-market trades."
        else:
            signal, note = "neutral", "Low conviction; minimal transaction volume."

        # Cluster buy: 3+ insiders buying within any 30-day window
        buy_dates = sorted(t["date"] for t in transactions if t["transaction_type"] == "P")
        cluster_buy = False
        if len(buy_dates) >= 3:
            for i in range(len(buy_dates) - 2):
                d0 = datetime.strptime(buy_dates[i],   "%Y-%m-%d")
                d2 = datetime.strptime(buy_dates[i+2], "%Y-%m-%d")
                if (d2 - d0).days <= 30:
                    cluster_buy = True
                    break

        out = {
            "ticker":                ticker.upper(),
            "filings_last_90_days":  len(form4_filings),
            "transactions_found":    len(transactions),
            "net_flow_usd":          int(net_flow_usd),
            "net_weighted_score":    round(net_weight, 2),
            "buy_sell_ratio":        round(purchases / sales, 2) if sales > 0 else (float("inf") if purchases > 0 else 0),
            "cluster_buy_signal":    cluster_buy,
            "insider_signal":        signal,
            "signal_note":           note,
            "classified_transactions": transactions[:10],
            "data_source":           "SEC EDGAR Form 4 (data.sec.gov submissions API + XML)",
            "_elapsed_ms":           round((time.time() - t0) * 1000),
        }
        log_fetch("SEC EDGAR Form4", ticker, cached=False, elapsed_ms=out["_elapsed_ms"])
        cache_set(ck, out)
        return out

    except Exception as e:
        out = {"error": str(e), "ticker": ticker, "data_source": "SEC EDGAR Form 4"}
        logger.error(f"get_edgar_form4({ticker}): {e}")
        return out


def get_edgar_filings(ticker: str, form_type: str = "8-K", limit: int = 5) -> dict:
    """
    Fetch recent SEC filings of a given type (8-K, 10-K, 10-Q, DEF 14A, etc.).
    Flags 8-K filings as potential material events.
    """
    ck = cache_key("edgar_filings", ticker, form_type)
    hit = cache_get(ck)
    if hit:
        log_fetch("SEC EDGAR Filings", ticker, cached=True)
        return hit

    t0 = time.time()

    try:
        cik = _get_cik(ticker)
        if not cik:
            return {"error": f"CIK not found for {ticker}", "ticker": ticker,
                    "data_source": "SEC EDGAR Submissions"}

        subs   = _get_submissions(cik)
        recent = subs.get("filings", {}).get("recent", {})

        forms  = recent.get("form", [])
        dates  = recent.get("filingDate", [])
        accs   = recent.get("accessionNumber", [])
        descs  = recent.get("primaryDocDescription", [None] * len(forms))

        matched = []
        for f, d, a, desc in zip(forms, dates, accs, descs):
            if form_type == "*" or f == form_type:
                accn_nd = a.replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nd}/{a}-index.htm"
                matched.append({
                    "filed":      d,
                    "form_type":  f,
                    "description": desc or f,
                    "accession_no": a,
                    "url":        url,
                    "is_material_event": f == "8-K",
                })
                if len(matched) >= limit:
                    break

        out = {
            "ticker":           ticker.upper(),
            "form_type":        form_type,
            "recent_filings":   matched,
            "filing_count":     len(matched),
            "data_source":      "SEC EDGAR Submissions API (data.sec.gov)",
            "_elapsed_ms":      round((time.time() - t0) * 1000),
        }
        log_fetch("SEC EDGAR Filings", ticker, cached=False, elapsed_ms=out["_elapsed_ms"])
        cache_set(ck, out)
        return out

    except Exception as e:
        out = {"error": str(e), "ticker": ticker, "data_source": "SEC EDGAR Submissions"}
        logger.error(f"get_edgar_filings({ticker}): {e}")
        return out


# Alias: get_edgar_form4 is the Form 4 insider transaction source
get_insider_activity = get_edgar_form4
