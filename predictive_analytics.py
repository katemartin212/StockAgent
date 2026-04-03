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
    "tnx":         "10Y Treasury Yield",
    "dxy":         "US Dollar Index",
    "vix":         "VIX Volatility",
    "sector":      "Sector ETF",
    "market":      "S&P 500 (Beta)",
    "momentum":    "4-Week Momentum",
    "reversal":    "1-Week Reversal",
    "value_spread": "Value/Growth Spread (IWD/IWF)",
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


def _ols(X: np.ndarray, y: np.ndarray, weights: np.ndarray = None):
    """
    Weighted Ridge regression with cross-validated L2 penalty and analytic SEs.
    X must be pre-normalized. weights are per-observation (unnormalized); None = uniform.
    Ridge alpha is selected via GCV (sklearn RidgeCV). SEs use the sandwich
    covariance  σ² · (X'WX + αI)⁻¹ · X'WX · (X'WX + αI)⁻¹  which reduces to
    OLS when α → 0 and weights are uniform.

    Returns (coefs, p_values, r_squared, ci_lo, ci_hi) or None.
    ci_lo / ci_hi are 95% CI lower/upper bounds on each slope coefficient.
    """
    from scipy import stats as sp
    from sklearn.linear_model import RidgeCV as _RidgeCV

    n, p = X.shape
    df = n - p - 1
    if df <= 0:
        return None

    if weights is None:
        weights = np.ones(n)
    # Normalize so weights sum to n (keeps RSS scale comparable to unweighted OLS)
    w = weights * (n / weights.sum())

    # Cross-validate Ridge penalty via GCV
    alphas = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    try:
        rcv = _RidgeCV(alphas=alphas, fit_intercept=True)
        rcv.fit(X, y, sample_weight=w)
        alpha = float(rcv.alpha_)
    except Exception:
        alpha = 1.0  # safe fallback

    # Augmented design matrix [1 | X]; penalty block excludes intercept
    Xc = np.column_stack([np.ones(n), X])
    P  = np.zeros((p + 1, p + 1))
    P[1:, 1:] = np.eye(p) * alpha

    try:
        XtWX = Xc.T @ (w[:, None] * Xc)   # avoids forming diag(w)
        A     = XtWX + P
        A_inv = np.linalg.inv(A)
        beta  = A_inv @ (Xc.T @ (w * y))

        resid  = y - Xc @ beta
        rss    = float((w * resid ** 2).sum())
        sigma2 = rss / df

        # Sandwich covariance: σ² · A⁻¹ · X'WX · A⁻¹
        var_beta = sigma2 * (A_inv @ XtWX @ A_inv)
        se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))

        t_stat = beta / np.where(se > 0, se, 1e-10)
        p_vals = 2 * (1 - sp.t.cdf(np.abs(t_stat), df=df))

        # Weighted R²
        y_wbar = float((w * y).sum()) / n
        ss_tot = float((w * (y - y_wbar) ** 2).sum())
        r2     = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0

        t_crit = sp.t.ppf(0.975, df=df)
        ci_lo  = beta - t_crit * se
        ci_hi  = beta + t_crit * se

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
            "iwd":    "IWD",   # Russell 1000 Value
            "iwf":    "IWF",   # Russell 1000 Growth
        }

        # Fetch all series in parallel
        with ThreadPoolExecutor(max_workers=9) as ex:
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

        # Momentum: 4-week trailing return, lagged 1 week (avoids look-ahead)
        mom = price_df["target"].pct_change(4).shift(1)
        ret_df["momentum"] = mom

        # Short-term reversal: 1-week lagged return (captures mean-reversion)
        ret_df["reversal"] = price_df["target"].pct_change(1).shift(1)

        # Value/growth spread: return of IWD/IWF ratio (value-vs-growth rotation)
        if "iwd" in price_df.columns and "iwf" in price_df.columns:
            ret_df["value_spread"] = (price_df["iwd"] / price_df["iwf"]).pct_change()

        # Explicit factor order: iwd/iwf are inputs to value_spread, not direct factors
        _FACTOR_ORDER = ["tnx", "dxy", "vix", "sector", "market", "momentum", "reversal", "value_spread"]
        factor_keys = [k for k in _FACTOR_ORDER if k in ret_df.columns]
        combined = ret_df[["target"] + factor_keys].dropna()
        if len(combined) < 52:
            return {"error": f"Insufficient aligned data after NA drop: {len(combined)} weeks"}

        y = combined["target"].values
        X_raw = combined[factor_keys].values
        mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_norm = (X_raw - mu) / sigma

        # Exponential decay: λ = 0.98/week, most recent observation weighted highest
        n_obs = len(combined)
        decay_weights = 0.98 ** np.arange(n_obs - 1, -1, -1)

        result = _ols(X_norm, y, weights=decay_weights)
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
            elif k == "reversal":
                col = price_df["target"].pct_change(1).dropna()
            elif k == "value_spread":
                if "iwd" in price_df.columns and "iwf" in price_df.columns:
                    col = (price_df["iwd"] / price_df["iwf"]).pct_change().dropna()
                else:
                    continue
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
                elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
                    # Newer yfinance versions return calendar as a wide DataFrame
                    # where "Earnings Date" is a column, not a row label
                    raw_dates = cal["Earnings Date"].dropna().tolist()
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
            # Recency weight: recent 4 quarters count double.
            # Magnitude weight: big beats (>5% surprise) count 1.5×,
            #   big misses (<-5% surprise) count 0.5× (tail events carry less signal).
            weighted_beats = 0.0
            total_weight   = 0.0
            for i, q in enumerate(quarterly_history):
                recency_mult = 2 if i < 4 else 1
                if q["beat"]:
                    mag_mult = 1.5 if q["surprise_pct"] > 5 else 1.0
                else:
                    mag_mult = 0.5 if q["surprise_pct"] < -5 else 1.0
                w = recency_mult * mag_mult
                if q["beat"]:
                    weighted_beats += w
                total_weight += w
            historical_beat_score = round(weighted_beats / total_weight * 100) if total_weight else 50

        # ── EPS estimate revision momentum ────────────────────────────────────
        revision_momentum_score = 50
        try:
            eps_trend = t.eps_trend
            if eps_trend is not None and not eps_trend.empty:
                # Structure (confirmed): index=periods (0q, +1q, 0y, +1y),
                # columns=time-ago (current, 7daysAgo, 30daysAgo, 60daysAgo, 90daysAgo).
                # Use current-quarter row; compare "current" vs "60daysAgo" columns.
                row = "0q" if "0q" in eps_trend.index else eps_trend.index[0]
                cur = float(eps_trend.loc[row, "current"])   if "current"   in eps_trend.columns else None
                ago = float(eps_trend.loc[row, "60daysAgo"]) if "60daysAgo" in eps_trend.columns else None
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

        # ── Sector read-through: sector ETF 4-week momentum vs 26-week z-score ──
        # Positive z (ETF outperforming recent avg) → bullish readthrough; negative → headwind.
        sector_readthrough_score = 50
        sector_etf_used = None
        try:
            sector_name = info.get("sector") or ""
            sector_etf  = SECTOR_ETF_MAP.get(sector_name, "")
            if sector_etf:
                sector_etf_used = sector_etf
                etf_prices = _fetch_weekly(sector_etf, "1y")
                if not etf_prices.empty:
                    ret4w = etf_prices.pct_change(4).dropna()
                    if len(ret4w) >= 27:
                        cur_ret4w = float(ret4w.iloc[-1])
                        baseline  = ret4w.iloc[-27:-1]   # prior 26 weeks — no look-ahead
                        m = float(baseline.mean())
                        s = float(baseline.std()) or 1.0
                        z = (cur_ret4w - m) / s
                        # z → score: 50 at z=0; ±2σ maps to [15, 85]
                        sector_readthrough_score = int(round(max(15.0, min(85.0, 50.0 + z * 17.5))))
        except Exception:
            pass

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
                "sector_etf":         sector_etf_used,
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

def get_scenario_analysis(ticker: str,
                           behavioral_inputs: dict | None = None,
                           peer_medians: dict | None = None) -> dict:
    """
    Three-stage DCF scenario model: bear / base / bull.

    The DCF fundamentals (revenue paths, FCF series, implied prices) are cached
    for 4 hours keyed only on ticker so the expensive data-fetch happens once.
    Probability adjustments and narrative adjustment consume behavioral_inputs
    fresh on every call (not cached) so they reflect the current analysis run.

    behavioral_inputs (optional) dict:
        divergence_score          float 0–100  (HIGH = overhyped, LOW = contrarian)
        macro_score               float 0–100  (50 = neutral; <30 = macro headwind)
        insider_signal            str  "strongly_bullish"|"bullish"|"neutral"|
                                       "bearish"|"strongly_bearish"
        earnings_surprise_probability  int 0–100
        sentiment_zscore          float

    peer_medians (optional) dict:
        ev_revenue   float   sector peer median EV/Revenue
        ev_ebitda    float   sector peer median EV/EBITDA
    """
    DCF_TTL = CACHE_TTL           # 4 h — fundamentals change slowly
    ck_dcf  = cache_key(f"pred_scenario_dcf_{ticker}")

    # ── Try DCF cache ─────────────────────────────────────────────────────────
    dcf = cache_get(ck_dcf, DCF_TTL)
    if dcf is None:
        dcf = _compute_dcf_core(ticker, peer_medians)
        if not dcf.get("error"):
            cache_set(ck_dcf, dcf)

    if dcf.get("error"):
        return dcf

    return _apply_behavioral(dcf, behavioral_inputs or {})


def _compute_dcf_core(ticker: str, peer_medians: dict | None = None) -> dict:
    """
    Fetch all fundamental data and compute the three-scenario DCF for ticker.
    Returns a 'dcf' dict that is cached for 4 hours; no behavioral adjustments.
    """
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        info = t.info or {}

        # ── Basics ────────────────────────────────────────────────────────────
        price  = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        if price <= 0:
            hist  = t.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        if price <= 0:
            return {"error": "Cannot determine current price"}

        shares     = float(info.get("sharesOutstanding") or 0)
        total_debt = float(info.get("totalDebt") or 0)
        total_cash = float(info.get("totalCash") or 0)
        net_cash   = total_cash - total_debt
        if shares <= 0:
            return {"error": "Missing share count"}

        analyst_target = float(info.get("targetMeanPrice") or 0) or None
        sector_name    = info.get("sector") or ""
        is_financial   = sector_name in ("Financial Services", "Financial")

        # ── TTM revenue ───────────────────────────────────────────────────────
        ttm_rev = float(info.get("totalRevenue") or 0)
        try:
            qs_stmt = t.quarterly_income_stmt
            if qs_stmt is not None and not qs_stmt.empty and "Total Revenue" in qs_stmt.index:
                if ttm_rev <= 0:
                    ttm_rev = float(qs_stmt.loc["Total Revenue"].iloc[:4].sum())
        except Exception:
            qs_stmt = None
        pre_revenue = ttm_rev <= 0

        # ── Margin inputs ─────────────────────────────────────────────────────
        gross_margin  = float(info.get("grossMargins") or 0)
        ebitda_margin = float(info.get("ebitdaMargins") or 0)
        beta          = max(0.5, min(2.5, float(info.get("beta") or 1.0)))

        # ── Income statement: operating income and gross profit ────────────────
        gross_profit_ttm  = gross_margin * ttm_rev
        operating_income  = 0.0
        try:
            inc = t.income_stmt
            if inc is not None and not inc.empty:
                for lbl in ("Operating Income", "EBIT", "Operating Profit"):
                    if lbl in inc.index:
                        operating_income = float(inc.loc[lbl].iloc[0])
                        break
                for lbl in ("Gross Profit",):
                    if lbl in inc.index:
                        gross_profit_ttm = float(inc.loc[lbl].iloc[0])
                        break
        except Exception:
            pass

        opex_ratio = 0.0  # SG&A + R&D + D&A as % of revenue
        if ttm_rev > 0:
            opex_ratio = max(0.0, (gross_profit_ttm - operating_income) / ttm_rev)
            if opex_ratio == 0 and ebitda_margin > 0:
                opex_ratio = max(0.0, gross_margin - ebitda_margin)

        # ── Cashflow: SBC and capex ───────────────────────────────────────────
        sbc_pct   = 0.0
        capex_pct = 0.0
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                for lbl in ("Stock Based Compensation", "StockBasedCompensation"):
                    if lbl in cf.index:
                        sbc_pct = abs(float(cf.loc[lbl].iloc[0])) / ttm_rev if ttm_rev > 0 else 0.0
                        break
                for lbl in ("Capital Expenditure", "CapitalExpenditures",
                             "Purchase of Property Plant and Equipment"):
                    if lbl in cf.index:
                        capex_pct = abs(float(cf.loc[lbl].iloc[0])) / ttm_rev if ttm_rev > 0 else 0.0
                        break
        except Exception:
            pass

        # ── Quarterly revenue history: 2Y CAGR + growth distribution ──────────
        rev_cagr_2y     = 0.0
        hist_p50_growth = 0.08
        hist_p75_growth = 0.12
        try:
            if qs_stmt is not None and not qs_stmt.empty and "Total Revenue" in qs_stmt.index:
                rev_s = qs_stmt.loc["Total Revenue"].dropna().sort_index()
                if len(rev_s) >= 8:
                    ttm_now = float(rev_s.iloc[-4:].sum())
                    ttm_2y  = float(rev_s.iloc[-8:-4].sum())
                    if ttm_2y > 0 and ttm_now > 0:
                        rev_cagr_2y = (ttm_now / ttm_2y) ** 0.5 - 1.0
                elif len(rev_s) >= 5:
                    old_q = float(rev_s.iloc[-5])
                    new_q = float(rev_s.iloc[-1])
                    if old_q > 0 and new_q > 0:
                        rev_cagr_2y = (new_q / old_q) - 1.0
                # Historical YoY quarterly growth distribution
                if len(rev_s) >= 8:
                    yoy = [(float(rev_s.iloc[i]) / float(rev_s.iloc[i - 4]) - 1.0)
                           for i in range(4, len(rev_s))
                           if float(rev_s.iloc[i - 4]) > 0]
                    if yoy:
                        hist_p50_growth = float(np.median(yoy))
                        hist_p75_growth = float(np.percentile(yoy, 75))
        except Exception:
            pass
        if rev_cagr_2y == 0.0:
            rev_cagr_2y = float(info.get("revenueGrowth") or 0) or 0.08

        # ── Consensus revenue estimates (years 1–2) ───────────────────────────
        # yfinance revenue_estimate rows are indexed by period strings:
        #   "0q" / "+1q" = quarterly;  "0y" / "+1y" = annual
        # We prefer annual rows. If only quarterly rows exist, annualise (×4)
        # but only if the resulting figure is plausible (> 0.9 × TTM revenue).
        # If consensus Y1 < 0.9 × TTM (decline not expected), discard it —
        # likely a stale or fiscal-year-timing artefact.
        rev_est_y1      = None
        rev_est_y2      = None
        using_consensus = False
        try:
            re_df = t.revenue_estimate
            if re_df is not None and not re_df.empty and "avg" in re_df.columns:
                rows    = [str(r) for r in re_df.index.tolist()]
                # Separate annual ("y") and quarterly ("q") rows
                ann_rows = [r for r in rows if "y" in r and "q" not in r]
                q_rows   = [r for r in rows if "q" in r]
                if ann_rows:
                    v1 = float(re_df.loc[ann_rows[0], "avg"])
                    v2 = float(re_df.loc[ann_rows[1], "avg"]) if len(ann_rows) > 1 else 0.0
                elif q_rows:
                    # Quarterly: annualise. Only accept if result > 0.9 × TTM.
                    v1 = float(re_df.loc[q_rows[0], "avg"]) * 4
                    v2 = 0.0
                else:
                    v1 = v2 = 0.0
                # Sanity gate: reject if below 90% of TTM (unexpected decline
                # is more likely a timing/fiscal-year artefact than reality)
                if v1 > 0 and (ttm_rev <= 0 or v1 >= ttm_rev * 0.90):
                    rev_est_y1      = v1
                    using_consensus = True
                if v2 > 0 and v2 >= (rev_est_y1 or 0) * 0.80:
                    rev_est_y2 = v2
        except Exception:
            pass
        if rev_est_y1 is None and ttm_rev > 0:
            rev_est_y1 = ttm_rev * (1 + rev_cagr_2y)
        if rev_est_y2 is None and rev_est_y1 is not None:
            rev_est_y2 = rev_est_y1 * (1 + rev_cagr_2y)

        # Consensus-implied Y1→Y2 growth rate (anchors mid-term base projection)
        # This is more reliable than rev_cagr_2y for years 3-5 because it reflects
        # what analysts expect for the near-term trajectory, not the historical CAGR.
        if using_consensus and rev_est_y2 and rev_est_y1 and rev_est_y1 > 0:
            consensus_g_fwd = rev_est_y2 / rev_est_y1 - 1.0
        else:
            consensus_g_fwd = rev_cagr_2y

        # ── 10-year Treasury rate ─────────────────────────────────────────────
        rate_10y = 4.5  # % — default if fetch fails
        try:
            tnx = yf.download("^TNX", period="5d", interval="1d",
                              progress=False, auto_adjust=True)
            if not tnx.empty:
                col = tnx["Close"] if "Close" in tnx.columns else tnx.iloc[:, 0]
                rate_10y = float(col.dropna().iloc[-1])
        except Exception:
            pass

        erp              = max(4.5, min(8.0, 4.5 + (beta - 1.0) * 1.5))
        discount_rate    = (rate_10y + erp) / 100.0

        # ── Terminal FCF multiple (based on gross margin quality) ─────────────
        tm_base   = 18.0 if gross_margin > 0.60 else (14.0 if gross_margin > 0.40 else 10.0)
        term_mult = {"bear": tm_base * 0.80, "base": tm_base, "bull": tm_base * 1.20}

        # ── Revenue path builder (10 years) ───────────────────────────────────
        N = 10

        def _rev_path(y1_mult, y2_mult, mid_g, terminal_g):
            if pre_revenue or rev_est_y1 is None:
                return [0.0] * N
            rev = [0.0] * (N + 1)
            rev[0] = ttm_rev
            rev[1] = max(0.0, rev_est_y1 * y1_mult)
            rev[2] = max(0.0, rev_est_y2 * y2_mult)
            for y in range(3, 6):
                rev[y] = rev[y - 1] * (1 + mid_g)
            for y in range(6, N + 1):
                fade   = (y - 5) / (N - 5)
                g      = mid_g * (1 - fade) + terminal_g * fade
                rev[y] = rev[y - 1] * (1 + g)
            return [max(0.0, rev[y]) for y in range(1, N + 1)]

        # Growth rates for years 3–5 in each scenario.
        # Base uses the consensus-implied forward growth (Y1→Y2), capped at 35%,
        # so cyclical CAGR spikes (MU 196%) and hypergrowth (NVDA 73%) don't
        # produce astronomically large year-5 revenues.
        base_mid = min(max(consensus_g_fwd, 0.0), 0.35)
        bear_mid = min(hist_p50_growth, 0.05)
        # Bull must always be MORE growth than base in years 3–5.
        # Floor at base_mid × 1.20 so the ordering bear < base < bull holds even
        # when hist_p75_growth is at its default (0.12) because yfinance returned
        # fewer than 8 quarters of quarterly data for this ticker.
        bull_mid = max(base_mid * 1.20, min(max(hist_p75_growth, 0.0), 0.50))

        rev_bear = _rev_path(0.92, 0.92, bear_mid, 0.025)
        rev_base = _rev_path(1.00, 1.00, base_mid,  0.030)
        rev_bull = _rev_path(1.08, 1.08, bull_mid,  0.035)

        # ── FCF margin path builder ───────────────────────────────────────────
        # FCF_t = Rev_t × (gross_margin_t − opex_t − capex_t − sbc_t)
        # bear: GM −150bps/yr×3, opex grows at 95% of rev, capex +15%, SBC flat
        # base: GM flat, opex proportional (100%), capex flat, SBC flat
        # bull: GM +100bps/yr×3, opex grows at 80% of rev, capex −10%, SBC −50bps/yr

        def _fcf_margins(scenario, rev_path):
            cap = capex_pct * (1.15 if scenario == "bear" else
                               0.90 if scenario == "bull" else 1.00)
            lev = 0.95 if scenario == "bear" else 0.80 if scenario == "bull" else 1.00
            gm, cur_opex, margins = gross_margin, opex_ratio, []
            prev_rev = ttm_rev
            for i in range(N):
                yr      = i + 1
                cur_rev = rev_path[i] if rev_path[i] > 0 else prev_rev
                # Gross margin evolution
                if scenario == "bear":
                    gm_yr = max(0.0, gross_margin - min(yr, 3) * 0.015)
                elif scenario == "bull":
                    gm_yr = min(1.0, gross_margin + min(yr, 3) * 0.010)
                else:
                    gm_yr = gross_margin
                # Operating leverage: opex grows at lev × rev_growth
                if prev_rev > 0 and cur_rev > prev_rev:
                    g_rev   = cur_rev / prev_rev - 1.0
                    cur_opex = (cur_opex * prev_rev * (1 + g_rev * lev)) / cur_rev
                # SBC
                sbc_yr = (max(0.0, sbc_pct - yr * 0.005) if scenario == "bull"
                          else sbc_pct)
                margins.append(gm_yr - cur_opex - cap - sbc_yr)
                prev_rev = cur_rev
            return margins

        def _fcf(rev_path, margin_path):
            return [rev_path[i] * margin_path[i] for i in range(N)]

        fcf_bear = _fcf(rev_bear, _fcf_margins("bear", rev_bear))
        fcf_base = _fcf(rev_base, _fcf_margins("base", rev_base))
        fcf_bull = _fcf(rev_bull, _fcf_margins("bull", rev_bull))

        # ── Financial-company guard ───────────────────────────────────────────
        # Banks and insurers have gross_margin ≈ 0 in yfinance (revenue = net
        # interest income, no COGS), so FCF = 0 and the DCF is meaningless.
        # Return None for all prices; the flag surface covers the explanation.
        dcf_not_applicable = is_financial

        # ── Deep-negative-FCF guard ───────────────────────────────────────────
        # Companies burning >50% of revenue as FCF (e.g. pre-profit hypergrowth
        # or quantum computing startups) produce negative DCF equity values that
        # floor at $0.01 — misleading rather than informative.
        # Detect by checking the BASE FCF margin in year 1.
        fm_base_check = _fcf_margins("base", rev_base)
        deeply_negative_fcf = (not pre_revenue and not dcf_not_applicable
                                and len(fm_base_check) > 0
                                and fm_base_check[0] < -0.50)
        if deeply_negative_fcf:
            dcf_not_applicable = True

        # ── DCF valuation ─────────────────────────────────────────────────────
        def _dcf(fcf_series, tm, nc):
            if pre_revenue or dcf_not_applicable:
                return None
            pv    = sum(f / (1 + discount_rate) ** (i + 1) for i, f in enumerate(fcf_series))
            fcf_t = fcf_series[-1]
            if fcf_t <= 0:
                fcf_t = max((f for f in fcf_series if f > 0), default=0.0) * 0.5
            tv    = fcf_t * tm
            pv   += tv / (1 + discount_rate) ** N
            equity = pv + nc
            return max(round(equity / shares, 2), 0.01) if equity > 0 else None

        bear_p = _dcf(fcf_bear, term_mult["bear"], net_cash)
        base_p = _dcf(fcf_base, term_mult["base"], net_cash)
        bull_p = _dcf(fcf_bull, term_mult["bull"], net_cash)

        def _ret(p):
            return round((p - price) / price * 100, 1) if (p and price > 0) else None

        # ── Implied multiples cross-check ─────────────────────────────────────
        def _impl_mult(p_scen, rev_y2, fcf_margin_y2):
            if p_scen is None or rev_y2 <= 0:
                return None, None
            ev        = p_scen * shares + total_debt - total_cash
            ev_rev    = round(ev / rev_y2, 2) if rev_y2 > 0 else None
            ebitda_y2 = rev_y2 * (fcf_margin_y2 + capex_pct + sbc_pct)
            ev_ebitda = round(ev / ebitda_y2, 1) if ebitda_y2 > 0 else None
            return ev_rev, ev_ebitda

        fm_bear = _fcf_margins("bear", rev_bear)
        fm_base = _fcf_margins("base", rev_base)
        fm_bull = _fcf_margins("bull", rev_bull)

        bear_ev_rev, bear_ev_ebitda = _impl_mult(bear_p, rev_bear[1], fm_bear[1])
        base_ev_rev, base_ev_ebitda = _impl_mult(base_p, rev_base[1], fm_base[1])
        bull_ev_rev, bull_ev_ebitda = _impl_mult(bull_p, rev_bull[1], fm_bull[1])

        # Peer median multiples
        pm_ev_rev = pm_ev_ebitda = None
        pm_source = "not available"
        if peer_medians:
            pm_ev_rev    = peer_medians.get("ev_revenue")
            pm_ev_ebitda = peer_medians.get("ev_ebitda")
            pm_source    = "peer median"
        else:
            try:
                from data_sources.comps_data import SECTOR_ETF as _SE
                etf = _SE.get(sector_name)
                if etf:
                    ei = yf.Ticker(etf).info or {}
                    pm_ev_rev    = ei.get("enterpriseToRevenue")
                    pm_ev_ebitda = ei.get("enterpriseToEbitda")
                    pm_source    = f"{etf} (sector ETF proxy)"
            except Exception:
                pass

        # ── Comps flags ───────────────────────────────────────────────────────
        flags = []
        if is_financial:
            flags.append(
                "Financial company — EV/Revenue is not meaningful for banks/insurers. "
                "DCF implied prices not computed; peer cross-check should use P/B or P/E."
            )
        if pre_revenue:
            flags.append(
                "Pre-revenue company — DCF model requires positive TTM revenue; "
                "implied prices not computed. Valuation requires stage-appropriate methods."
            )
        if deeply_negative_fcf:
            flags.append(
                f"Deeply negative FCF ({round(fm_base_check[0]*100,0):.0f}% margin yr 1) — "
                "company is pre-profitability; DCF cannot produce reliable implied prices. "
                "Consider EV/Revenue or milestone-based valuation instead."
            )
        if pm_ev_rev and bull_ev_rev and bull_ev_rev > 2 * pm_ev_rev:
            flags.append(
                f"Bull case implies EV/Revenue {bull_ev_rev}× "
                f"— more than 2× peer median ({pm_ev_rev}×). "
                "Requires exceptional execution."
            )
        if pm_ev_rev and bear_ev_rev and bear_ev_rev < 0.5 * pm_ev_rev:
            flags.append(
                f"Bear case implies EV/Revenue {bear_ev_rev}× "
                f"— below 0.5× peer median ({pm_ev_rev}×). Implies distress pricing."
            )

        # ── Auto-surface limitations ──────────────────────────────────────────
        active_limitations = []
        if 0 < gross_margin < 0.15:
            active_limitations.append(
                "Limitation 10 — Low gross margin: EV/EBITDA or P/E would be a more "
                "appropriate anchor than EV/Revenue for this company's profitability profile."
            )
        if is_financial:
            active_limitations.append(
                "Limitation 10 — Financial sector: debt is inventory, not financing. "
                "Standard DCF / EV metrics don't apply to banks/insurers."
            )
        try:
            tnx_hist = yf.download("^TNX", period="2y", interval="1wk",
                                   progress=False, auto_adjust=True)
            if not tnx_hist.empty:
                col = (tnx_hist["Close"] if "Close" in tnx_hist.columns
                       else tnx_hist.iloc[:, 0]).dropna()
                if len(col) >= 52:
                    delta = abs(float(col.iloc[-1]) - float(col.iloc[0]))
                    if delta > 1.5:
                        active_limitations.append(
                            f"Limitation 11 — Rate environment: 10Y Treasury has moved "
                            f"{delta:.1f}pp over 2 years; discount rate and historical "
                            "multiple distributions span different rate regimes."
                        )
        except Exception:
            pass

        # ── Net cash at year 3 (scenario-dependent, informational) ────────────
        nc_y3 = {
            s: round((net_cash + sum(fcf[:3])) / 1e9, 2)
            for s, fcf in (("bear", fcf_bear), ("base", fcf_base), ("bull", fcf_bull))
        }

        # ── Year 1–3 projection summaries ─────────────────────────────────────
        def _yr_summary(rev_path, fcf_series):
            out = []
            for i in range(min(3, N)):
                r = rev_path[i]
                f = fcf_series[i]
                out.append({
                    "year":           i + 1,
                    "revenue":        f"${r / 1e9:.1f}B" if r >= 1e9 else f"${r / 1e6:.0f}M",
                    "revenue_b":      round(r / 1e9, 2),
                    "fcf":            (f"+${f / 1e9:.1f}B" if f >= 0 else f"-${abs(f) / 1e9:.1f}B")
                                      if abs(f) >= 1e9 else
                                      (f"+${f / 1e6:.0f}M" if f >= 0 else f"-${abs(f) / 1e6:.0f}M"),
                    "fcf_b":          round(f / 1e9, 2),
                    "fcf_margin_pct": round(f / r * 100, 1) if r > 0 else None,
                })
            return out

        return {
            # meta
            "current_price":          round(price, 2),
            "analyst_target":         round(analyst_target, 2) if analyst_target else None,
            "sector":                 sector_name,
            "is_financial":           is_financial,
            "pre_revenue":            pre_revenue,
            "deeply_negative_fcf":    deeply_negative_fcf,
            "dcf_not_applicable":     dcf_not_applicable,
            # DCF inputs
            "discount_rate_pct":      round(rate_10y + erp, 2),
            "rate_10y_pct":           round(rate_10y, 2),
            "erp_pct":                round(erp, 2),
            "beta":                   round(beta, 2),
            "terminal_multiple_base": round(tm_base, 1),
            "gross_margin_pct":       round(gross_margin * 100, 1),
            "sbc_pct":                round(sbc_pct * 100, 2),
            "capex_pct":              round(capex_pct * 100, 2),
            "rev_cagr_2y_pct":        round(rev_cagr_2y * 100, 1),
            "using_consensus":        using_consensus,
            # DCF outputs
            "bear_p":  bear_p,  "base_p":  base_p,  "bull_p":  bull_p,
            "bear_ret": _ret(bear_p), "base_ret": _ret(base_p), "bull_ret": _ret(bull_p),
            # year projections
            "yr_bear": _yr_summary(rev_bear, fcf_bear),
            "yr_base": _yr_summary(rev_base, fcf_base),
            "yr_bull": _yr_summary(rev_bull, fcf_bull),
            # key drivers
            "bear_driver": (
                f"Year 1–2 revenue 8% below consensus; gross margin −150bps/yr×3; "
                f"growth fades to {round(bear_mid * 100, 1)}% by year 5"
            ),
            "base_driver": (
                f"Revenue at consensus (Y1 ${round(rev_est_y1/1e9,1)}B, "
                f"+{round((rev_est_y1/ttm_rev - 1)*100,0):.0f}% vs TTM); margins flat"
                if (using_consensus and ttm_rev and ttm_rev > 0) else
                f"Revenue at 2Y CAGR ({round(rev_cagr_2y * 100, 1)}%); margins flat"
            ),
            "bull_driver": (
                f"Year 1–2 revenue 8% above consensus; gross margin +100bps/yr×3; "
                f"growth reaches {round(bull_mid * 100, 1)}% by year 5"
            ),
            # comps
            "bear_ev_rev": bear_ev_rev, "base_ev_rev": base_ev_rev, "bull_ev_rev": bull_ev_rev,
            "bear_ev_ebitda": bear_ev_ebitda, "base_ev_ebitda": base_ev_ebitda,
            "bull_ev_ebitda": bull_ev_ebitda,
            "pm_ev_rev": pm_ev_rev, "pm_ev_ebitda": pm_ev_ebitda, "pm_source": pm_source,
            # net cash evolution
            "net_cash_y3_bear": nc_y3["bear"],
            "net_cash_y3_base": nc_y3["base"],
            "net_cash_y3_bull": nc_y3["bull"],
            # flags
            "flags":              flags,
            "active_limitations": active_limitations,
            "error": None,
        }

    except Exception as e:
        logger.error(f"_compute_dcf_core({ticker}): {e}", exc_info=True)
        return {"error": str(e)}


def _apply_behavioral(dcf: dict, bi: dict) -> dict:
    """
    Layer probability adjustments and narrative adjustment onto the cached DCF core.
    bi keys (all optional, defaults to neutral if absent):
        divergence_score          float 0–100  (HIGH = overhyped)
        macro_score               float 0–100  (50 = neutral; <30 = headwind)
        insider_signal            str
        earnings_surprise_probability  int 0–100
        sentiment_zscore          float
    """
    bear_p = dcf["bear_p"]
    base_p = dcf["base_p"]
    bull_p = dcf["bull_p"]

    div_score   = float(bi.get("divergence_score", 50))
    macro_score = float(bi.get("macro_score",       50))
    insider     = str(bi.get("insider_signal",      "neutral"))
    earn_prob   = int(bi.get("earnings_surprise_probability", 50))
    sent_z      = float(bi.get("sentiment_zscore",  0.0))

    # ── Earnings surprise adjustment to base revenue estimate ─────────────────
    earn_adj = +0.03 if earn_prob > 65 else (-0.03 if earn_prob < 35 else 0.0)
    if earn_adj != 0.0 and base_p is not None:
        base_p = round(base_p * (1.0 + earn_adj), 2)

    # ── Narrative (behavioral finance) adjustment to base case only ───────────
    # (divergence_score − 50) / 50 × 10%: overhyped → discount, contrarian → premium
    narr_adj_frac = (div_score - 50.0) / 50.0 * 0.10
    base_p_dcf    = base_p
    base_p_adj    = round(base_p * (1.0 - narr_adj_frac), 2) if base_p else None

    def _ret(p):
        cur = dcf["current_price"]
        return round((p - cur) / cur * 100, 1) if (p and cur > 0) else None

    # ── Scenario probabilities from fundamental inputs ─────────────────────────
    bear_prob, prob_log_bear = 25, []
    if div_score > 70:
        bear_prob += 10;  prob_log_bear.append("divergence_score > 70 (+10%)")
    if macro_score < 30:
        bear_prob += 8;   prob_log_bear.append("macro_score < 30 (+8%)")
    if insider in ("bearish", "strongly_bearish"):
        bear_prob += 7;   prob_log_bear.append(f"insider {insider} (+7%)")
    if earn_prob < 35:
        bear_prob += 5;   prob_log_bear.append("earnings_prob < 35 (+5%)")
    if insider == "strongly_bullish":
        bear_prob -= 8;   prob_log_bear.append("insider strongly_bullish (−8%)")
    if earn_prob > 70:
        bear_prob -= 6;   prob_log_bear.append("earnings_prob > 70 (−6%)")
    if sent_z < -1.5:
        bear_prob -= 5;   prob_log_bear.append(f"sentiment z={sent_z:.1f} contrarian (−5%)")
    bear_prob = max(10, min(50, bear_prob))

    bull_prob, prob_log_bull = 25, []
    if earn_prob > 70:
        bull_prob += 10;  prob_log_bull.append("earnings_prob > 70 (+10%)")
    if insider == "strongly_bullish":
        bull_prob += 8;   prob_log_bull.append("insider strongly_bullish (+8%)")
    if sent_z < -1.5:
        bull_prob += 7;   prob_log_bull.append(f"sentiment z={sent_z:.1f} contrarian (+7%)")
    if div_score < 30:
        bull_prob += 5;   prob_log_bull.append("divergence_score < 30 (+5%)")
    if div_score > 70:
        bull_prob -= 10;  prob_log_bull.append("divergence_score > 70 (−10%)")
    if macro_score < 20:
        bull_prob -= 8;   prob_log_bull.append("macro_score < 20 (−8%)")
    bull_prob = max(10, min(50, bull_prob))
    base_prob = max(0, 100 - bear_prob - bull_prob)

    # ── Probability-weighted targets ──────────────────────────────────────────
    def _wt(bear, base, bull):
        if None in (bear, base, bull):
            return None
        return round(bear * (bear_prob / 100) +
                     base * (base_prob / 100) +
                     bull * (bull_prob / 100), 2)

    weighted           = _wt(bear_p, base_p, bull_p)
    weighted_narrative = _wt(bear_p, base_p_adj, bull_p)

    return {
        "current_price":    dcf["current_price"],
        "analyst_target":   dcf["analyst_target"],
        "weighted_target":  weighted_narrative,      # headline: narrative-adjusted
        "weighted_target_dcf": weighted,             # pure DCF weighted target
        # DCF context
        "discount_rate_pct":      dcf["discount_rate_pct"],
        "rate_10y_pct":           dcf["rate_10y_pct"],
        "erp_pct":                dcf["erp_pct"],
        "beta":                   dcf["beta"],
        "terminal_multiple_base": dcf["terminal_multiple_base"],
        "gross_margin_pct":       dcf["gross_margin_pct"],
        "sbc_pct":                dcf["sbc_pct"],
        "capex_pct":              dcf["capex_pct"],
        "rev_cagr_2y_pct":        dcf["rev_cagr_2y_pct"],
        "using_consensus":        dcf["using_consensus"],
        "pre_revenue":         dcf["pre_revenue"],
        "deeply_negative_fcf": dcf["deeply_negative_fcf"],
        "dcf_not_applicable":  dcf["dcf_not_applicable"],
        "is_financial":        dcf["is_financial"],
        "scenarios": {
            "bear": {
                "price":               bear_p,
                "return_pct":          _ret(bear_p),
                "key_driver":          dcf["bear_driver"],
                "probability":         bear_prob,
                "probability_drivers": prob_log_bear,
                "year_projections":    dcf["yr_bear"],
                "implied_ev_revenue":  dcf["bear_ev_rev"],
                "implied_ev_ebitda":   dcf["bear_ev_ebitda"],
                "terminal_multiple":   round(dcf["terminal_multiple_base"] * 0.80, 1),
                "net_cash_yr3_b":      dcf["net_cash_y3_bear"],
            },
            "base": {
                "price":               base_p_adj,   # narrative-adjusted is the headline
                "price_dcf":           base_p_dcf,
                "narrative_adj_pct":   round(-narr_adj_frac * 100, 1),
                "return_pct":          _ret(base_p_adj),
                "key_driver":          dcf["base_driver"] + (
                    f"; earnings surprise adj {'+' if earn_adj > 0 else ''}"
                    f"{round(earn_adj * 100, 0):.0f}%" if earn_adj else ""
                ) + (
                    f"; narrative {'discount' if narr_adj_frac > 0 else 'premium'} "
                    f"{abs(round(narr_adj_frac * 100, 1))}%"
                ),
                "probability":         base_prob,
                "probability_drivers": [],
                "year_projections":    dcf["yr_base"],
                "implied_ev_revenue":  dcf["base_ev_rev"],
                "implied_ev_ebitda":   dcf["base_ev_ebitda"],
                "terminal_multiple":   round(dcf["terminal_multiple_base"], 1),
                "net_cash_yr3_b":      dcf["net_cash_y3_base"],
            },
            "bull": {
                "price":               bull_p,
                "return_pct":          _ret(bull_p),
                "key_driver":          dcf["bull_driver"],
                "probability":         bull_prob,
                "probability_drivers": prob_log_bull,
                "year_projections":    dcf["yr_bull"],
                "implied_ev_revenue":  dcf["bull_ev_rev"],
                "implied_ev_ebitda":   dcf["bull_ev_ebitda"],
                "terminal_multiple":   round(dcf["terminal_multiple_base"] * 1.20, 1),
                "net_cash_yr3_b":      dcf["net_cash_y3_bull"],
            },
        },
        "peer_comparison": {
            "source":              dcf["pm_source"],
            "peer_ev_revenue":     dcf["pm_ev_rev"],
            "peer_ev_ebitda":      dcf["pm_ev_ebitda"],
            "base_implied_ev_rev": dcf["base_ev_rev"],
            "base_implied_ev_ebitda": dcf["base_ev_ebitda"],
        },
        "narrative_adjustment": {
            "divergence_score": div_score,
            "adjustment_pct":   round(-narr_adj_frac * 100, 1),
            "base_dcf":         base_p_dcf,
            "base_adjusted":    base_p_adj,
            "explanation": (
                f"Divergence {div_score:.0f}/100 → "
                f"{'narrative premium discount' if narr_adj_frac > 0 else 'contrarian premium'} "
                f"of {abs(round(narr_adj_frac * 100, 1))}% on DCF base"
            ),
        },
        "probability_inputs": {
            "divergence_score":              div_score,
            "macro_score":                   macro_score,
            "insider_signal":                insider,
            "earnings_surprise_probability": earn_prob,
            "sentiment_zscore":              sent_z,
        },
        "flags":              dcf["flags"],
        "active_limitations": dcf["active_limitations"],
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SENTIMENT MEAN REVERSION
# ══════════════════════════════════════════════════════════════════════════════

def get_sentiment_mean_reversion(ticker: str, sector: str = None) -> dict:
    ck = cache_key(f"pred_sentiment_mr_{ticker}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    try:
        from pytrends.request import TrendReq
        from datetime import datetime, timedelta
        time.sleep(1.2)  # avoid rate limiting
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=0)
        end = datetime.now()
        start = end - timedelta(days=730)
        tf = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
        pytrends.build_payload([ticker], cat=0, timeframe=tf, geo="US")
        df = pytrends.interest_over_time()

        if df.empty or ticker not in df.columns:
            return {"error": f"No Google Trends data for {ticker}"}

        scores = df[ticker].dropna()
        if len(scores) < 12:
            return {"error": f"Insufficient Trends history: {len(scores)} weeks"}

        # 6-month (26-week) statistics for Trends signal
        # FIX: baseline mean/std computed on scores.iloc[:-2] to exclude the
        # two most-recent incomplete weeks — prevents look-ahead contamination
        baseline = scores.iloc[:-2] if len(scores) > 28 else scores
        last26   = baseline.tail(26)
        mean26   = float(last26.mean())
        std26    = float(last26.std()) or 1.0

        # Current reading = average of latest 4 weeks (these ARE current, not baseline)
        cur4w    = float(scores.tail(4).mean())
        trends_z = round((cur4w - mean26) / std26, 2)
        pct_vs   = round((cur4w - mean26) / mean26 * 100, 1) if mean26 > 0 else 0.0

        # ── Reddit sentiment signal ────────────────────────────────────────────
        # weighted_score (0-100, neutral=50) → z-scale with σ=16.7 so that
        # ±25 from neutral maps to ±1.5σ, matching the Trends signal threshold.
        # Skipped if Reddit finds low coverage (score defaults to 50, not meaningful).
        reddit_z       = None
        reddit_wscore  = None
        reddit_coverage = False
        try:
            from data_sources.reddit_sentiment import get_reddit_sentiment
            rd = get_reddit_sentiment(ticker, sector=sector)
            if not rd.get("low_coverage") and rd.get("weighted_score") is not None:
                reddit_wscore   = int(rd["weighted_score"])
                reddit_z        = round((reddit_wscore - 50) / 16.7, 2)
                reddit_coverage = True
        except Exception:
            pass

        # ── Composite z-score (50/50 when both sources available) ─────────────
        if reddit_coverage and reddit_z is not None:
            z           = round((trends_z + reddit_z) / 2, 2)
            sources_used = "Google Trends + Reddit"
        else:
            z           = trends_z
            sources_used = "Google Trends"

        if z > 1.5:
            signal = "elevated"
            signal_text = (
                f"Elevated sentiment — contrarian caution signal. "
                f"Retail search interest is {abs(pct_vs):.0f}% above 6-month average "
                f"({sources_used})."
            )
        elif z < -1.5:
            signal = "depressed"
            signal_text = (
                f"Depressed sentiment — contrarian opportunity signal. "
                f"Retail search interest is {abs(pct_vs):.0f}% below 6-month average "
                f"({sources_used})."
            )
        else:
            signal = "neutral"
            signal_text = (
                f"Sentiment within normal range (composite z-score: {z}, {sources_used}). "
                f"No strong contrarian signal."
            )

        # ── Historical correlation: sentiment → subsequent 4-week return ──────
        historical_correlation = None
        correlation_validated  = False
        correlation_note       = "Correlation data unavailable."
        try:
            prices = _fetch_weekly(ticker, "2y")
            if not prices.empty:
                # Normalize both indexes to week-ending Sunday to align Trends (Sun) with yfinance (Mon)
                scores_aligned = scores.copy()
                scores_aligned.index = pd.to_datetime(scores_aligned.index).tz_localize(None).to_period("W").to_timestamp("W")
                prices_aligned = prices.copy()
                prices_aligned.index = pd.to_datetime(prices_aligned.index).tz_localize(None).to_period("W").to_timestamp("W")
                fwd_returns = prices_aligned.pct_change(4).shift(-4)
                aligned = pd.DataFrame({"sent": scores_aligned, "fwd": fwd_returns}).dropna()
                if len(aligned) >= 24:
                    from scipy.stats import spearmanr as _spearmanr
                    corr, _ = _spearmanr(aligned["sent"].values, aligned["fwd"].values)
                    corr = float(corr)
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
            "z_score":                z,             # composite (Trends+Reddit) or Trends-only
            "trends_z":               trends_z,
            "reddit_z":               reddit_z,
            "reddit_weighted_score":  reddit_wscore,
            "sources_used":           sources_used,
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

def run_all_predictive(ticker: str, sector: str = None,
                        behavioral_inputs: dict | None = None) -> dict:
    """
    Two-phase execution so scenario analysis has full behavioral context.

    Phase 1 (parallel): factor attribution, earnings surprise probability,
        sentiment mean reversion — these are independent of each other.
    Phase 2 (sequential): scenario analysis — consumes Phase 1 outputs
        (earnings_probability, sentiment_zscore) merged with any
        behavioral_inputs from master_signal (divergence, macro, insider).
    """
    # ── Phase 1: independent models ──────────────────────────────────────────
    p1_tasks = {
        "factor_attribution":   (get_factor_attribution,           (ticker, sector)),
        "earnings_probability": (get_earnings_surprise_probability, (ticker,)),
        "sentiment_mr":         (get_sentiment_mean_reversion,      (ticker, sector)),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        fmap = {ex.submit(fn, *args): key for key, (fn, args) in p1_tasks.items()}
        try:
            for fut in as_completed(fmap, timeout=60):
                key = fmap[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    results[key] = {"error": str(e)}
        except FuturesTimeout:
            for fut, key in fmap.items():
                if key not in results:
                    results[key] = {"error": "timed out"}

    # ── Enrich behavioral_inputs with Phase 1 results ─────────────────────────
    bi = dict(behavioral_inputs or {})
    ep = results.get("earnings_probability") or {}
    sm = results.get("sentiment_mr") or {}
    if not ep.get("error"):
        bi.setdefault("earnings_surprise_probability", ep.get("probability", 50))
    if not sm.get("error"):
        bi.setdefault("sentiment_zscore", sm.get("z", 0.0))

    # ── Phase 2: scenario with full behavioral context ─────────────────────────
    try:
        results["scenario"] = get_scenario_analysis(ticker, behavioral_inputs=bi)
    except Exception as e:
        results["scenario"] = {"error": str(e)}

    return results
