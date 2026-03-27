#!/usr/bin/env python3
"""
predictive_analytics.py — Four forward-looking analytics functions for the Predict tab.

    get_factor_attribution(ticker, sector=None)      → dict
    get_earnings_surprise_probability(ticker)         → dict
    get_scenario_analysis(ticker)                     → dict
    get_sentiment_mean_reversion(ticker)              → dict
    run_all_predictive(ticker, sector=None)           → dict (runs all 4 in parallel)

All functions:
- Use a 4-hour cache TTL (predictive data changes slowly)
- Fail gracefully with {"error": "..."} rather than raising
- Require at least 52 weeks of history or return an insufficient-data state
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

from data_sources._cache import cache_get, cache_set, cache_key

RNG = np.random.default_rng(42)

logger = logging.getLogger("stock_agent")

CACHE_TTL = 4 * 3600  # 4 hours

SECTOR_ETF_MAP = {
    "Technology":             "XLK",
    "Communication Services": "XLC",
    "Healthcare":             "XLV",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Energy":                 "XLE",
    "Real Estate":            "XLRE",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Utilities":              "XLU",
}

FACTOR_LABELS = {
    "tnx":      "10Y Treasury Yield",
    "dxy":      "US Dollar Index",
    "vix":      "VIX Volatility",
    "sector":   "Sector ETF",
    "market":   "S&P 500 (Beta)",
    "momentum": "4-Week Momentum",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_weekly(symbol: str, period: str = "2y") -> pd.Series:
    """Fetch weekly adjusted close prices for one symbol. Silent on error."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, interval="1wk", auto_adjust=True)
        if hist.empty:
            return pd.Series(dtype=float, name=symbol)
        return hist["Close"].rename(symbol)
    except Exception:
        return pd.Series(dtype=float, name=symbol)


def _ols(X: np.ndarray, y: np.ndarray):
    """
    OLS with scipy t-distribution p-values and 95% confidence intervals.
    X must be pre-normalized. Returns (coefs, p_values, r_squared, ci_lo, ci_hi) or None.
    ci_lo / ci_hi are 95% CI lower/upper bounds on each slope coefficient.
    """
    from scipy import stats as sp
    n, p = X.shape
    df = n - p - 1
    if df <= 0:
        return None
    Xc = np.column_stack([np.ones(n), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
        resid = y - Xc @ beta
        s2 = float(resid @ resid) / df
        XtXi = np.linalg.inv(Xc.T @ Xc)
        se = np.sqrt(s2 * np.diag(XtXi))
        t_stat = beta / se
        p_vals = 2 * (1 - sp.t.cdf(np.abs(t_stat), df=df))
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        t_crit = sp.t.ppf(0.975, df=df)
        ci_lo = beta - t_crit * se
        ci_hi = beta + t_crit * se
        return beta[1:], p_vals[1:], float(r2), ci_lo[1:], ci_hi[1:]
    except (np.linalg.LinAlgError, ValueError):
        return None


def _pct_rank(series: pd.Series, value: float) -> float:
    valid = series.dropna()
    return float((valid <= value).mean() * 100) if len(valid) > 0 else 50.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. FACTOR ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def get_factor_attribution(ticker: str, sector: str = None) -> dict:
    ck = cache_key(f"pred_factor_{ticker}_{sector or ''}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    try:
        sector_etf = SECTOR_ETF_MAP.get(sector or "", "SPY")
        factor_symbols = {
            "tnx":    "^TNX",
            "dxy":    "DX-Y.NYB",
            "vix":    "^VIX",
            "sector": sector_etf,
            "market": "^GSPC",
        }

        # Fetch all series in parallel
        with ThreadPoolExecutor(max_workers=7) as ex:
            futures = {ex.submit(_fetch_weekly, sym, "2y"): name
                       for name, sym in factor_symbols.items()}
            futures[ex.submit(_fetch_weekly, ticker, "2y")] = "target"
            price_map = {}
            try:
                for fut in as_completed(futures, timeout=30):
                    price_map[futures[fut]] = fut.result()
            except FuturesTimeout:
                for fut, name in futures.items():
                    if name not in price_map:
                        price_map[name] = pd.Series(dtype=float)

        if "target" not in price_map or price_map["target"].empty:
            return {"error": "No price data returned for ticker"}

        # Normalize all series to timezone-naive UTC dates, then resample to weekly
        # to handle mixed timezones (e.g. ^TNX/^VIX use Chicago, equities use NY)
        for k in list(price_map.keys()):
            s = price_map[k]
            if s.empty:
                continue
            if s.index.tz is not None:
                s = s.tz_convert("UTC").tz_localize(None)
            price_map[k] = s.resample("W-FRI").last()

        price_df = pd.DataFrame(price_map).ffill().dropna(how="any")
        if len(price_df) < 52:
            return {"error": f"Insufficient history: {len(price_df)} weeks (need ≥52)"}

        ret_df = price_df.pct_change().dropna()

        # Momentum: 4-week trailing return of the ticker itself, lagged 1 week
        mom = price_df["target"].pct_change(4).shift(1)
        ret_df["momentum"] = mom

        factor_keys = [k for k in list(factor_symbols.keys()) + ["momentum"]
                       if k in ret_df.columns]
        combined = ret_df[["target"] + factor_keys].dropna()
        if len(combined) < 52:
            return {"error": f"Insufficient aligned data after NA drop: {len(combined)} weeks"}

        y = combined["target"].values
        X_raw = combined[factor_keys].values
        mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_norm = (X_raw - mu) / sigma

        result = _ols(X_norm, y)
        if result is None:
            return {"error": "OLS regression failed — singular matrix"}
        coefs, p_vals, r2, ci_lo, ci_hi = result

        # Benjamini-Hochberg FDR correction on factor p-values
        from statsmodels.stats.multitest import multipletests
        _, fdr_pvals, _, _ = multipletests(p_vals, method="fdr_bh")

        factors = []
        for i, k in enumerate(factor_keys):
            factors.append({
                "name":           k,
                "label":          FACTOR_LABELS.get(k, k),
                "coefficient":    round(float(coefs[i]), 4),
                "p_value":        round(float(p_vals[i]), 4),
                "p_value_fdr":    round(float(fdr_pvals[i]), 4),
                "ci_95_lo":       round(float(ci_lo[i]), 4),
                "ci_95_hi":       round(float(ci_hi[i]), 4),
                "significant":    bool(fdr_pvals[i] < 0.05),      # FDR-corrected significance
                "significant_raw": bool(p_vals[i] < 0.05),        # Uncorrected, for reference
                "direction":      "positive" if coefs[i] > 0 else "negative",
            })
        factors.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

        # Current conditions vs historical range
        # FIX: use col.iloc[:-1] so we never include the current (incomplete) week
        # in the range calculation — prevents look-ahead bias in percentile rank
        current_conditions = []
        for fc in factors:
            k = fc["name"]
            if k == "momentum":
                col = price_df["target"].pct_change(4).dropna()
            elif k in price_df.columns:
                col = price_df[k].dropna()
            else:
                continue
            if len(col) < 2:
                continue
            # Current value = last observation; range = all prior observations
            cur   = float(col.iloc[-1])
            prior = col.iloc[:-1]                                   # exclude current week
            p_rank = _pct_rank(prior, cur)
            headwind = (fc["direction"] == "positive" and p_rank < 35) or \
                       (fc["direction"] == "negative" and p_rank > 65)
            current_conditions.append({
                "factor":      k,
                "label":       fc["label"],
                "current":     round(cur, 3),
                "pct_rank":    round(p_rank, 0),
                "signal":      "headwind" if headwind else "tailwind",
                "coefficient": fc["coefficient"],
                "significant": fc["significant"],
            })

        out = {
            "factors":            factors,
            "r_squared":          round(r2, 4),
            "r_squared_pct":      round(r2 * 100, 1),
            "n_weeks":            int(len(combined)),
            "sector_etf":         sector_etf,
            "current_conditions": current_conditions,
            "fdr_note":           (
                "Factor significance uses Benjamini-Hochberg FDR correction "
                f"({sum(1 for f in factors if f['significant'])}/{len(factors)} factors significant after correction)."
            ),
            "error":              None,
        }
        cache_set(ck, out)
        return out

    except Exception as e:
        logger.error(f"get_factor_attribution({ticker}): {e}", exc_info=True)
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 2. EARNINGS SURPRISE PROBABILITY
# ══════════════════════════════════════════════════════════════════════════════

def get_earnings_surprise_probability(ticker: str) -> dict:
    ck = cache_key(f"pred_earnings_{ticker}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}

        # ── Next earnings date ────────────────────────────────────────────────
        next_earnings_date = None
        days_to_earnings = None
        try:
            cal = t.calendar
            if cal is not None:
                raw_dates = None
                if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    raw_dates = cal.loc["Earnings Date"].tolist()
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    raw_dates = cal["Earnings Date"]
                    if not isinstance(raw_dates, list):
                        raw_dates = [raw_dates]
                if raw_dates:
                    for d in raw_dates:
                        ned = pd.to_datetime(d, errors="coerce")
                        if ned and pd.notna(ned) and ned > pd.Timestamp.now():
                            next_earnings_date = ned.strftime("%Y-%m-%d")
                            days_to_earnings = (ned.date() - datetime.now().date()).days
                            break
        except Exception:
            pass

        if not next_earnings_date:
            try:
                ed = t.earnings_dates
                if ed is not None and not ed.empty:
                    future = ed[ed.index > pd.Timestamp.now()]
                    if not future.empty:
                        ned = future.index[-1]
                        next_earnings_date = ned.strftime("%Y-%m-%d")
                        days_to_earnings = (ned.date() - datetime.now().date()).days
            except Exception:
                pass

        # ── Historical beat rate (last 8 quarters) ────────────────────────────
        quarterly_history = []
        historical_beat_score = 50
        try:
            eh = None
            for attr in ("earnings_history", "get_earnings_history"):
                try:
                    eh = getattr(t, attr)
                    if callable(eh):
                        eh = eh()
                    if eh is not None and (not isinstance(eh, pd.DataFrame) or not eh.empty):
                        break
                except Exception:
                    pass

            if isinstance(eh, pd.DataFrame) and not eh.empty:
                today = datetime.now().date()
                for idx, row in eh.iterrows():
                    # FIX: only include past earnings events — no look-ahead
                    event_date = pd.to_datetime(idx, errors="coerce")
                    if event_date is not None and pd.notna(event_date):
                        if event_date.date() >= today:
                            continue
                    actual   = row.get("epsActual")   or row.get("EPS Actual")
                    estimate = row.get("epsEstimate") or row.get("EPS Estimate")
                    if actual is None or estimate is None:
                        continue
                    actual, estimate = float(actual), float(estimate)
                    if pd.isna(actual) or pd.isna(estimate) or estimate == 0:
                        continue
                    beat = actual >= estimate
                    quarterly_history.append({
                        "quarter":      str(idx)[:7],
                        "actual":       round(actual, 3),
                        "estimate":     round(estimate, 3),
                        "beat":         beat,
                        "surprise_pct": round((actual - estimate) / abs(estimate) * 100, 1),
                    })
                # Keep last 8 quarters for display
                quarterly_history = quarterly_history[:8]
        except Exception:
            pass

        if quarterly_history:
            # Recent 4 quarters count double
            weighted_beats = sum((2 if i < 4 else 1) for i, q in enumerate(quarterly_history) if q["beat"])
            total_weight   = sum((2 if i < 4 else 1) for i in range(len(quarterly_history)))
            historical_beat_score = round(weighted_beats / total_weight * 100) if total_weight else 50

        # ── EPS estimate revision momentum ────────────────────────────────────
        revision_momentum_score = 50
        try:
            eps_trend = t.eps_trend
            if eps_trend is not None and not eps_trend.empty:
                col = "0q" if "0q" in eps_trend.columns else eps_trend.columns[0]
                cur  = float(eps_trend.loc["current",   col]) if "current"   in eps_trend.index else None
                ago  = float(eps_trend.loc["60daysAgo", col]) if "60daysAgo" in eps_trend.index else None
                if cur is not None and ago is not None and ago != 0:
                    chg = (cur - ago) / abs(ago) * 100
                    if   chg > 5:   revision_momentum_score = min(100, 80 + int(chg * 0.4))
                    elif chg > 1:   revision_momentum_score = 75
                    elif chg > -1:  revision_momentum_score = 50
                    elif chg > -5:  revision_momentum_score = 25
                    else:           revision_momentum_score = max(0, 20 - int(abs(chg) * 0.4))
        except Exception:
            pass

        # ── Analyst sentiment (upgrades vs downgrades, last 30 days) ──────────
        analyst_sentiment_score = 50
        try:
            ud = t.upgrades_downgrades
            if ud is not None and not ud.empty:
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
                recent = ud[ud.index > cutoff]
                if not recent.empty and "ToGrade" in recent.columns:
                    grades = recent["ToGrade"].str.lower()
                    ups   = grades.str.contains(r"buy|overweight|outperform|positive|accumulate", na=False).sum()
                    downs = grades.str.contains(r"sell|underweight|underperform|negative|reduce", na=False).sum()
                    total = ups + downs
                    if total > 0:
                        analyst_sentiment_score = round(ups / total * 100)
        except Exception:
            pass

        # Sector read-through: no real-time peer data → neutral 50
        sector_readthrough_score = 50

        probability = round(
            historical_beat_score   * 0.35 +
            revision_momentum_score * 0.35 +
            analyst_sentiment_score * 0.15 +
            sector_readthrough_score * 0.15
        )

        # Bootstrap 95% CI on probability using beat history
        prob_ci_lo, prob_ci_hi = None, None
        if quarterly_history:
            outcomes = np.array([1 if q["beat"] else 0 for q in quarterly_history])
            boot_probs = []
            for _ in range(1000):
                sample = RNG.choice(outcomes, size=len(outcomes), replace=True)
                boot_beat = round(float(sample.mean()) * 100)
                boot_p = round(
                    boot_beat            * 0.35 +
                    revision_momentum_score * 0.35 +
                    analyst_sentiment_score * 0.15 +
                    sector_readthrough_score * 0.15
                )
                boot_probs.append(boot_p)
            prob_ci_lo = int(np.percentile(boot_probs, 2.5))
            prob_ci_hi = int(np.percentile(boot_probs, 97.5))

        out = {
            "probability":        probability,
            "prob_ci_lo":         prob_ci_lo,
            "prob_ci_hi":         prob_ci_hi,
            "n_quarters":         len(quarterly_history),
            "next_earnings_date": next_earnings_date,
            "days_to_earnings":   days_to_earnings,
            "sub_scores": {
                "historical_beat":    historical_beat_score,
                "revision_momentum":  revision_momentum_score,
                "analyst_sentiment":  analyst_sentiment_score,
                "sector_readthrough": sector_readthrough_score,
            },
            "quarterly_history": quarterly_history,
            "error": None,
        }
        cache_set(ck, out)
        return out

    except Exception as e:
        logger.error(f"get_earnings_surprise_probability({ticker}): {e}", exc_info=True)
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def get_scenario_analysis(ticker: str) -> dict:
    ck = cache_key(f"pred_scenario_{ticker}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    try:
        import yfinance as yf
        t   = yf.Ticker(ticker)
        info = t.info or {}

        # ── Current price ─────────────────────────────────────────────────────
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        if price <= 0:
            hist = t.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
        if price <= 0:
            return {"error": "Cannot determine current price"}

        shares     = float(info.get("sharesOutstanding") or 0)
        total_debt = float(info.get("totalDebt") or 0)
        total_cash = float(info.get("totalCash") or 0)
        net_cash   = total_cash - total_debt  # positive = net cash

        # TTM revenue
        ttm_rev = float(info.get("totalRevenue") or 0)
        if ttm_rev <= 0:
            try:
                qs = t.quarterly_income_stmt
                if qs is not None and not qs.empty and "Total Revenue" in qs.index:
                    ttm_rev = float(qs.loc["Total Revenue"].iloc[:4].sum())
            except Exception:
                pass

        if ttm_rev <= 0 or shares <= 0:
            return {"error": "Missing revenue or share count — cannot model scenarios"}

        analyst_target = float(info.get("targetMeanPrice") or 0) or None

        # ── 2-year historical EV/Revenue range ───────────────────────────────
        price_hist = _fetch_weekly(ticker, "2y")
        ev_rev_hist = []
        if not price_hist.empty:
            for p_val in price_hist.values:
                if pd.isna(p_val) or p_val <= 0:
                    continue
                ev = p_val * shares + total_debt - total_cash
                ev_rev = ev / ttm_rev
                if 0 < ev_rev < 1000:
                    ev_rev_hist.append(ev_rev)

        if ev_rev_hist:
            arr = np.array(ev_rev_hist)
            cur_ev_rev = float(arr[-1])
            # FIX: exclude last 13 weeks from percentile range (one quarter)
            # so current valuation is benchmarked against prior history only
            hist_for_pct = arr[:-13] if len(arr) > 13 else arr[:-1]
            if len(hist_for_pct) < 4:
                hist_for_pct = arr
            ev25 = float(np.percentile(hist_for_pct, 25))
            ev75 = float(np.percentile(hist_for_pct, 75))
        else:
            cur_ev_rev = ((price * shares + total_debt - total_cash) / ttm_rev) if ttm_rev else None
            if cur_ev_rev is None:
                return {"error": "Cannot compute EV/Revenue"}
            ev25 = cur_ev_rev * 0.65
            ev75 = cur_ev_rev * 1.35

        # ── Revenue growth rate ───────────────────────────────────────────────
        rev_cagr = float(info.get("revenueGrowth") or 0)
        if rev_cagr == 0:
            try:
                qs = t.quarterly_income_stmt
                if qs is not None and not qs.empty and "Total Revenue" in qs.index:
                    rev_s = qs.loc["Total Revenue"].sort_index()
                    if len(rev_s) >= 5:
                        old, new = float(rev_s.iloc[0]), float(rev_s.iloc[4])
                        if old > 0 and new > 0:
                            rev_cagr = (new / old) - 1.0
            except Exception:
                pass
        if rev_cagr == 0:
            rev_cagr = 0.08  # fallback

        # ── Scenario implied prices ───────────────────────────────────────────
        def _implied_price(growth_rate, ev_rev_multiple):
            future_rev = ttm_rev * (1 + growth_rate)
            future_ev  = future_rev * ev_rev_multiple
            return max(round((future_ev + net_cash) / shares, 2), 0.01)

        def _ret(p):
            return round((p - price) / price * 100, 1) if price > 0 else 0.0

        bear_p = _implied_price(rev_cagr * 0.50, ev25)
        base_p = _implied_price(rev_cagr,         cur_ev_rev)
        bull_p = _implied_price(rev_cagr * 1.20,  ev75)

        weighted = round(bear_p * 0.25 + base_p * 0.50 + bull_p * 0.25, 2)

        out = {
            "current_price":      round(price, 2),
            "analyst_target":     round(analyst_target, 2) if analyst_target else None,
            "weighted_target":    weighted,
            "scenarios": {
                "bear": {
                    "price":      bear_p,
                    "return_pct": _ret(bear_p),
                    "key_driver": (
                        f"Revenue growth slows to {round(rev_cagr * 50, 1)}% "
                        f"and EV/Rev compresses to {round(ev25, 1)}× (25th pct of 2Y range)"
                    ),
                    "probability": 25,
                },
                "base": {
                    "price":      base_p,
                    "return_pct": _ret(base_p),
                    "key_driver": (
                        f"Revenue grows at trailing rate ({round(rev_cagr * 100, 1)}%) "
                        f"and multiple holds at {round(cur_ev_rev, 1)}× EV/Rev"
                    ),
                    "probability": 50,
                },
                "bull": {
                    "price":      bull_p,
                    "return_pct": _ret(bull_p),
                    "key_driver": (
                        f"Revenue growth accelerates to {round(rev_cagr * 120, 1)}% "
                        f"and EV/Rev re-rates to {round(ev75, 1)}× (75th pct of 2Y range)"
                    ),
                    "probability": 25,
                },
            },
            "ev_revenue_current": round(cur_ev_rev, 2),
            "ev_revenue_25th":    round(ev25, 2),
            "ev_revenue_75th":    round(ev75, 2),
            "rev_cagr_pct":       round(rev_cagr * 100, 1),
            "error": None,
        }
        cache_set(ck, out)
        return out

    except Exception as e:
        logger.error(f"get_scenario_analysis({ticker}): {e}", exc_info=True)
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. SENTIMENT MEAN REVERSION
# ══════════════════════════════════════════════════════════════════════════════

def get_sentiment_mean_reversion(ticker: str) -> dict:
    ck = cache_key(f"pred_sentiment_mr_{ticker}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    try:
        from pytrends.request import TrendReq
        time.sleep(1.2)  # avoid rate limiting
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=0)
        pytrends.build_payload([ticker], cat=0, timeframe="today 24-m", geo="US")
        df = pytrends.interest_over_time()

        if df.empty or ticker not in df.columns:
            return {"error": f"No Google Trends data for {ticker}"}

        scores = df[ticker].dropna()
        if len(scores) < 12:
            return {"error": f"Insufficient Trends history: {len(scores)} weeks"}

        # 6-month (26-week) statistics for signal
        # FIX: baseline mean/std computed on scores.iloc[:-2] to exclude the
        # two most-recent incomplete weeks — prevents look-ahead contamination
        baseline = scores.iloc[:-2] if len(scores) > 28 else scores
        last26   = baseline.tail(26)
        mean26   = float(last26.mean())
        std26    = float(last26.std()) or 1.0

        # Current reading = average of latest 4 weeks (these ARE current, not baseline)
        cur4w  = float(scores.tail(4).mean())
        z      = round((cur4w - mean26) / std26, 2)
        pct_vs = round((cur4w - mean26) / mean26 * 100, 1) if mean26 > 0 else 0.0

        if z > 1.5:
            signal = "elevated"
            signal_text = (
                f"Elevated sentiment — contrarian caution signal. "
                f"Retail search interest is {abs(pct_vs):.0f}% above 6-month average."
            )
        elif z < -1.5:
            signal = "depressed"
            signal_text = (
                f"Depressed sentiment — contrarian opportunity signal. "
                f"Retail search interest is {abs(pct_vs):.0f}% below 6-month average."
            )
        else:
            signal = "neutral"
            signal_text = f"Sentiment within normal range (z-score: {z}). No strong contrarian signal."

        # ── Historical correlation: sentiment → subsequent 4-week return ──────
        historical_correlation = None
        correlation_validated  = False
        correlation_note       = "Correlation data unavailable."
        try:
            prices = _fetch_weekly(ticker, "2y")
            if not prices.empty:
                fwd_returns = prices.pct_change(4).shift(-4)
                aligned = pd.DataFrame({"sent": scores, "fwd": fwd_returns}).dropna()
                if len(aligned) >= 24:
                    corr = float(aligned["sent"].corr(aligned["fwd"]))
                    historical_correlation = round(corr, 3)
                    correlation_validated  = abs(corr) > 0.2
                    if correlation_validated:
                        direction = "preceded underperformance" if corr < 0 else "preceded outperformance"
                        correlation_note = (
                            f"Historical correlation: {corr:+.2f} (validated signal) — "
                            f"elevated sentiment has historically {direction} for {ticker}."
                        )
                    else:
                        correlation_note = (
                            f"Historical correlation: {corr:+.2f} (weak) — "
                            f"limited predictive value for {ticker} historically."
                        )
        except Exception:
            pass

        # Serialize weekly series for chart (last 52 weeks)
        weeks_data = [
            {"week": str(idx.date()), "score": int(v)}
            for idx, v in scores.tail(52).items()
            if pd.notna(v)
        ]

        out = {
            "z_score":                z,
            "current_4w_avg":         round(cur4w, 1),
            "six_month_avg":          round(mean26, 1),
            "six_month_std":          round(std26, 1),
            "pct_above_avg":          pct_vs,
            "signal":                 signal,
            "signal_text":            signal_text,
            "weeks_data":             weeks_data,
            "historical_correlation": historical_correlation,
            "correlation_validated":  correlation_validated,
            "correlation_note":       correlation_note,
            "upper_threshold":        round(mean26 + 1.5 * std26, 1),
            "lower_threshold":        round(mean26 - 1.5 * std26, 1),
            "mean_line":              round(mean26, 1),
            "error": None,
        }
        cache_set(ck, out)
        return out

    except Exception as e:
        logger.error(f"get_sentiment_mean_reversion({ticker}): {e}", exc_info=True)
        return {"error": str(e)}


# ── Run all four in parallel ──────────────────────────────────────────────────

def run_all_predictive(ticker: str, sector: str = None) -> dict:
    """Run all 4 predictive functions in parallel. Returns structured dict."""
    tasks = {
        "factor_attribution":   (get_factor_attribution,           (ticker, sector)),
        "earnings_probability": (get_earnings_surprise_probability, (ticker,)),
        "scenario":             (get_scenario_analysis,             (ticker,)),
        "sentiment_mr":         (get_sentiment_mean_reversion,      (ticker,)),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        future_map = {ex.submit(fn, *args): key for key, (fn, args) in tasks.items()}
        try:
            for fut in as_completed(future_map, timeout=65):
                key = future_map[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    results[key] = {"error": str(e)}
        except FuturesTimeout:
            for fut, key in future_map.items():
                if key not in results:
                    results[key] = {"error": "timed out"}
    return results
