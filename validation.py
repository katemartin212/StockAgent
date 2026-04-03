#!/usr/bin/env python3
"""
validation.py — Walk-forward out-of-sample validation for predictive models.

    validate_factor_model(ticker, sector=None)   → dict
    validate_earnings_model(ticker)              → dict
    validate_sentiment_signal(ticker)            → dict
    run_full_validation(ticker, sector=None)     → dict

Confidence tiers:
    HIGH        = 3 / 3 models validated
    MEDIUM      = 2 / 3 models validated
    LOW         = 1 / 3 models validated
    UNVALIDATED = 0 / 3 models validated

Minimum data gates:
    Factor model:    ≥ 104 weeks of aligned returns
    Earnings model:  ≥ 6 historical earnings events with actual vs estimate
    Sentiment model: ≥ 24 aligned (sentiment, forward-return) weeks
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from scipy import stats as sp

from data_sources._cache import cache_get, cache_set, cache_key
from predictive._timeseries import align_weekly as _fetch_weekly, decay_weights as _decay_weights
from predictive._ridge import weighted_ridge as _ols
from predictive_analytics import SECTOR_ETF_MAP, FACTOR_LABELS

logger = logging.getLogger("stock_agent")

CACHE_TTL    = 6 * 3600   # 6 hours — validation is expensive
RNG          = np.random.default_rng(42)
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1. FACTOR MODEL — Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_factor_model(ticker: str, sector: str = None) -> dict:
    """
    Expanding-window walk-forward OLS validation.

    Protocol:
      - Minimum 104 weeks of aligned return data required.
      - Initial training window: first 52 weeks.
      - Each subsequent week: refit OLS on [0, t), predict week t.
      - Out-of-sample metrics: MAE, directional accuracy, IC (Spearman rank).
      - Validated if directional_accuracy > 52% AND |IC| > 0.04.

    Returns dict with keys: validated (bool), directional_accuracy, ic,
    mae, r2_oos, n_oos, n_train_final, confidence_note, error.
    """
    ck = cache_key(f"val_factor_{ticker}_{sector or ''}")
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

        with ThreadPoolExecutor(max_workers=9) as ex:
            futures = {ex.submit(_fetch_weekly, sym, "5y"): name
                       for name, sym in factor_symbols.items()}
            futures[ex.submit(_fetch_weekly, ticker, "5y")] = "target"
            price_map = {}
            try:
                for fut in as_completed(futures, timeout=40):
                    price_map[futures[fut]] = fut.result()
            except FuturesTimeout:
                for fut, name in futures.items():
                    if name not in price_map:
                        price_map[name] = pd.Series(dtype=float)

        if "target" not in price_map or price_map["target"].empty:
            return _val_error("factor", "No price data for ticker")

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
        ret_df   = price_df.pct_change().dropna()
        mom      = price_df["target"].pct_change(4).shift(1)
        ret_df["momentum"] = mom

        # Mirror production model: short-term reversal and value/growth spread
        ret_df["reversal"] = price_df["target"].pct_change(1).shift(1)
        if "iwd" in price_df.columns and "iwf" in price_df.columns:
            ret_df["value_spread"] = (price_df["iwd"] / price_df["iwf"]).pct_change()

        _FACTOR_ORDER = ["tnx", "dxy", "vix", "sector", "market", "momentum", "reversal", "value_spread"]
        factor_keys = [k for k in _FACTOR_ORDER if k in ret_df.columns]
        combined = ret_df[["target"] + factor_keys].dropna()

        if len(combined) < 104:
            return _val_error(
                "factor",
                f"Insufficient history: {len(combined)} weeks (need ≥104 for walk-forward validation)"
            )

        # Build a lagged prediction dataset:
        #   X_lag[t] = factor returns for week t  (known at end of week t)
        #   y_lag[t] = target return for week t+1 (to be predicted at week t+1)
        # This ensures we never use future information: X[t] predicts y[t+1].
        raw_y = combined["target"].values
        raw_X = combined[factor_keys].values
        X_lag = raw_X[:-1]          # weeks 0 … N-2 (predictor)
        y_lag = raw_y[1:]           # weeks 1 … N-1 (outcome to predict)
        n        = len(y_lag)
        min_train = 52

        actuals, predicted = [], []

        for t in range(min_train, n):
            # Train on lagged pairs [0, t) — no look-ahead
            X_tr, y_tr = X_lag[:t], y_lag[:t]
            mu, sigma  = X_tr.mean(axis=0), X_tr.std(axis=0)
            sigma[sigma == 0] = 1.0
            X_tr_norm = (X_tr - mu) / sigma

            # Predict week t+1 using X_lag[t] (week t factors — already known)
            X_te_norm = (X_lag[t] - mu) / sigma

            # Exponential decay: most recent training obs gets weight 1.0
            decay_weights_tr = _decay_weights(t)

            result = _ols(X_tr_norm, y_tr, weights=decay_weights_tr)
            if result is None:
                continue
            coefs = result[0]  # (coefs, p_vals, r2, ci_lo, ci_hi)
            pred = float(np.dot(coefs, X_te_norm))
            actuals.append(float(y_lag[t]))
            predicted.append(pred)

        if len(actuals) < 20:
            return _val_error("factor", f"Too few OOS predictions ({len(actuals)}) — validation inconclusive")

        actuals   = np.array(actuals)
        predicted = np.array(predicted)

        mae = float(np.mean(np.abs(actuals - predicted)))
        directional_accuracy = float(np.mean(np.sign(actuals) == np.sign(predicted)))

        # Wilson 95% CI on directional accuracy (proportion of correct sign predictions).
        # Works well at small n and near the 50% boundary — unlike the normal approximation.
        #   center = (p̂ + z²/2n) / (1 + z²/n)
        #   margin = z·√(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n)
        _n   = len(actuals)
        _k   = int(np.sum(np.sign(actuals) == np.sign(predicted)))
        _z   = 1.96
        _z2  = _z ** 2
        _den = 1.0 + _z2 / _n
        _ctr = (directional_accuracy + _z2 / (2 * _n)) / _den
        _mrg = (_z * np.sqrt(
            directional_accuracy * (1 - directional_accuracy) / _n + _z2 / (4 * _n ** 2)
        )) / _den
        da_ci_lo = float(max(0.0, _ctr - _mrg))
        da_ci_hi = float(min(1.0, _ctr + _mrg))

        # Information Coefficient: Spearman rank correlation
        ic_val, ic_pval = sp.spearmanr(predicted, actuals)
        ic = float(ic_val)

        # OOS R²
        ss_res = float(np.sum((actuals - predicted) ** 2))
        ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
        r2_oos = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Validated only if the 95% CI lower bound clears 50% (not just the point estimate)
        # and IC exceeds threshold. Prevents a 53% accuracy on 50 samples from passing.
        validated = da_ci_lo > 0.50 and abs(ic) > 0.04

        da_pct    = round(directional_accuracy * 100, 1)
        da_margin = round((da_ci_hi - da_ci_lo) / 2 * 100, 1)

        if validated:
            note = (
                f"Factor model passes walk-forward validation. "
                f"Directional accuracy {da_pct}% ±{da_margin}% [95% CI: "
                f"{round(da_ci_lo * 100, 1)}%–{round(da_ci_hi * 100, 1)}%], "
                f"IC {ic:+.3f} over {len(actuals)} OOS weeks."
            )
        else:
            reasons = []
            if da_ci_lo <= 0.50:
                reasons.append(
                    f"directional accuracy 95% CI [{round(da_ci_lo * 100, 1)}%–"
                    f"{round(da_ci_hi * 100, 1)}%] includes 50%"
                )
            if abs(ic) <= 0.04:
                reasons.append(f"|IC| {abs(ic):.3f} ≤ 0.04")
            note = f"Factor model does NOT pass validation: {'; '.join(reasons)}."

        out = {
            "validated":              validated,
            "directional_accuracy":   da_pct,
            "directional_accuracy_ci_lo": round(da_ci_lo * 100, 1),
            "directional_accuracy_ci_hi": round(da_ci_hi * 100, 1),
            "directional_accuracy_margin": da_margin,
            "ic":                     round(ic, 4),
            "ic_p_value":             round(float(ic_pval), 4),
            "mae":                    round(mae * 100, 4),
            "r2_oos":                 round(r2_oos, 4),
            "n_oos":                  len(actuals),
            "n_train_final":          n - 1,
            "confidence_note":        note,
            "error":                  None,
        }
        cache_set(ck, out)
        return out

    except Exception as e:
        logger.error(f"validate_factor_model({ticker}): {e}", exc_info=True)
        return _val_error("factor", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 2. EARNINGS MODEL — Leave-one-out cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_earnings_model(ticker: str) -> dict:
    """
    Leave-one-out cross-validation of the earnings beat probability model.

    For each historical earnings event i:
      - Compute sub-scores using all events EXCEPT event i.
      - Record predicted probability vs actual outcome (beat/miss).
    Metrics: Brier score, Brier Skill Score (vs naive 50% baseline).
    Validated if brier_skill_score > 0 and n_events ≥ 6.

    Brier score:     mean((p - o)²), lower = better, perfect = 0
    Brier skill:     1 - BS / BS_ref, positive = better than naive guess
    """
    ck = cache_key(f"val_earnings_{ticker}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    try:
        import yfinance as yf
        t     = yf.Ticker(ticker)
        today = datetime.now().date()
        events = []

        # Primary source: earnings_dates (contains full history — up to ~25 events)
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                for idx, row in ed.iterrows():
                    event_date = pd.to_datetime(idx, errors="coerce")
                    if event_date is None or pd.isna(event_date):
                        continue
                    if event_date.tz is not None:
                        event_date = event_date.tz_convert("UTC").tz_localize(None)
                    if event_date.date() >= today:
                        continue
                    estimate = row.get("EPS Estimate")
                    actual   = row.get("Reported EPS")
                    if estimate is None or actual is None:
                        continue
                    try:
                        estimate, actual = float(estimate), float(actual)
                    except (TypeError, ValueError):
                        continue
                    if pd.isna(estimate) or pd.isna(actual) or estimate == 0:
                        continue
                    beat = actual >= estimate
                    surprise_pct = (actual - estimate) / abs(estimate) * 100
                    events.append({"beat": beat, "surprise_pct": float(surprise_pct),
                                   "actual": actual, "estimate": estimate})
        except Exception:
            pass

        # Fallback: earnings_history (only last 4 quarters)
        if len(events) < 4:
            for attr in ("earnings_history", "get_earnings_history"):
                try:
                    eh = getattr(t, attr)
                    if callable(eh): eh = eh()
                    if eh is None or (isinstance(eh, pd.DataFrame) and eh.empty):
                        continue
                    for idx, row in eh.iterrows():
                        actual   = row.get("epsActual")   or row.get("EPS Actual")
                        estimate = row.get("epsEstimate") or row.get("EPS Estimate")
                        if actual is None or estimate is None: continue
                        try: actual, estimate = float(actual), float(estimate)
                        except: continue
                        if pd.isna(actual) or pd.isna(estimate) or estimate == 0: continue
                        event_date = pd.to_datetime(idx, errors="coerce")
                        if event_date is not None and pd.notna(event_date):
                            if event_date.date() >= today: continue
                        beat = actual >= estimate
                        surprise_pct = (actual - estimate) / abs(estimate) * 100
                        events.append({"beat": beat, "surprise_pct": float(surprise_pct),
                                       "actual": actual, "estimate": estimate})
                    break
                except Exception:
                    pass

        if not events:
            return _val_error("earnings", "No earnings history available")

        if len(events) < 6:
            return _val_error(
                "earnings",
                f"Insufficient earnings events: {len(events)} (need ≥6 for LOO validation)"
            )

        # LOO cross-validation
        probs, outcomes = [], []

        for i in range(len(events)):
            loo = [e for j, e in enumerate(events) if j != i]
            # Historical beat rate from remaining events
            beats_recent = sum(1 for j, e in enumerate(loo) if e["beat"] and j >= max(0, len(loo) - 4))
            weight_recent = sum(2 if j >= max(0, len(loo) - 4) else 1 for j in range(len(loo)))
            beats_weighted = sum(
                (2 if j >= max(0, len(loo) - 4) else 1)
                for j, e in enumerate(loo) if e["beat"]
            )
            hist_score = round(beats_weighted / weight_recent * 100) if weight_recent else 50

            # Revision momentum: use mean surprise of other events as proxy
            surprises     = [e["surprise_pct"] for e in loo]
            mean_surprise = float(np.mean(surprises)) if surprises else 0.0
            if   mean_surprise > 5:   rev_score = min(100, 80 + int(mean_surprise * 0.4))
            elif mean_surprise > 1:   rev_score = 75
            elif mean_surprise > -1:  rev_score = 50
            elif mean_surprise > -5:  rev_score = 25
            else:                     rev_score = max(0, 20 - int(abs(mean_surprise) * 0.4))

            # Analyst + sector neutral for LOO (no per-event override)
            prob = round(hist_score * 0.35 + rev_score * 0.35 + 50 * 0.15 + 50 * 0.15)
            probs.append(prob / 100.0)
            outcomes.append(1 if events[i]["beat"] else 0)

        probs, outcomes = np.array(probs), np.array(outcomes)

        brier_score  = float(np.mean((probs - outcomes) ** 2))
        brier_ref    = float(np.mean((0.5 * np.ones_like(outcomes) - outcomes) ** 2))
        brier_skill  = float(1.0 - brier_score / brier_ref) if brier_ref > 0 else 0.0
        avg_prob     = float(probs.mean())
        beat_rate    = float(outcomes.mean())
        calibration  = round(abs(avg_prob - beat_rate), 3)

        validated = brier_skill > 0 and len(events) >= 6

        if validated:
            note = (
                f"Earnings model passes LOO validation. "
                f"Brier Skill Score {brier_skill:+.3f} (positive = better than naive) "
                f"over {len(events)} events."
            )
        else:
            note = (
                f"Earnings model does NOT pass validation. "
                f"Brier Skill Score {brier_skill:+.3f} over {len(events)} events."
            )

        out = {
            "validated":      validated,
            "brier_score":    round(brier_score, 4),
            "brier_skill":    round(brier_skill, 4),
            "beat_rate":      round(beat_rate * 100, 1),
            "avg_predicted":  round(avg_prob * 100, 1),
            "calibration":    calibration,
            "n_events":       len(events),
            "confidence_note": note,
            "error":          None,
        }
        cache_set(ck, out)
        return out

    except Exception as e:
        logger.error(f"validate_earnings_model({ticker}): {e}", exc_info=True)
        return _val_error("earnings", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 3. SENTIMENT SIGNAL — Statistical significance test
# ══════════════════════════════════════════════════════════════════════════════

def validate_sentiment_signal(ticker: str) -> dict:
    """
    Tests whether extreme sentiment readings (|z| > 1.5) predict 4-week returns.

    For each week where sentiment z-score > 1.5 (elevated) or < -1.5 (depressed),
    record the subsequent 4-week return. Then:
      - t-test vs zero for each direction
      - Bootstrap 95% CI on mean forward return (1000 resamples, seed=42)
      - Annualised Sharpe of the full signal (elevated → short, depressed → long)

    Validated if |t-statistic| > 1.96 (p < 0.05) for either direction AND n_signals ≥ 8.
    """
    ck = cache_key(f"val_sentiment_{ticker}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    try:
        from pytrends.request import TrendReq
        from datetime import datetime, timedelta
        import time
        time.sleep(1.2)
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=0)
        end = datetime.now()
        start = end - timedelta(days=730)
        tf = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
        pytrends.build_payload([ticker], cat=0, timeframe=tf, geo="US")
        df_trends = pytrends.interest_over_time()

        if df_trends.empty or ticker not in df_trends.columns:
            return _val_error("sentiment", f"No Google Trends data for {ticker}")

        scores = df_trends[ticker].dropna()
        # Normalize to week-ending Sunday so it aligns with yfinance (Monday open → same ISO week)
        scores.index = pd.to_datetime(scores.index).tz_localize(None).to_period("W").to_timestamp("W")
        if len(scores) < 24:
            return _val_error("sentiment", f"Insufficient Trends history: {len(scores)} weeks (need ≥24)")

        prices = _fetch_weekly(ticker, "2y")
        if prices.empty:
            return _val_error("sentiment", "No price data for return calculation")
        prices.index = pd.to_datetime(prices.index).tz_localize(None).to_period("W").to_timestamp("W")

        # 4-week forward returns (shift -4 so we never look ahead when building signal)
        fwd_returns = prices.pct_change(4).shift(-4)
        aligned = pd.DataFrame({"sent": scores, "fwd": fwd_returns}).dropna()

        if len(aligned) < 24:
            return _val_error("sentiment", f"Insufficient aligned data: {len(aligned)} weeks (need ≥24)")

        # Rolling z-score: baseline = previous 26 weeks (no look-ahead)
        z_scores = []
        for i in range(len(aligned)):
            if i < 26:
                z_scores.append(float("nan"))
                continue
            # Use iloc[:i] — excludes current week
            window = aligned["sent"].iloc[i - 26:i]
            m, s   = float(window.mean()), float(window.std())
            s      = s if s > 0 else 1.0
            z_scores.append((aligned["sent"].iloc[i] - m) / s)

        aligned["z"] = z_scores
        aligned = aligned.dropna()

        elevated  = aligned[aligned["z"] >  1.5]["fwd"].values
        depressed = aligned[aligned["z"] < -1.5]["fwd"].values
        n_signals = len(elevated) + len(depressed)

        if n_signals < 8:
            return _val_error(
                "sentiment",
                f"Too few extreme signals: {n_signals} (need ≥8 for statistical validity)"
            )

        def _t_test(arr):
            if len(arr) < 3:
                return None, None, (None, None)
            t_stat, p_val = sp.ttest_1samp(arr, 0)
            # Bootstrap 95% CI
            boot_means = [float(RNG.choice(arr, size=len(arr), replace=True).mean())
                          for _ in range(1000)]
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
            return float(t_stat), float(p_val), (ci_lo, ci_hi)

        t_elev,  p_elev,  ci_elev  = _t_test(elevated)
        t_dep,   p_dep,   ci_dep   = _t_test(depressed)

        # Sharpe: elevated → expected negative return (short signal),
        #         depressed → expected positive return (long signal)
        # Combine into a trading signal: long depressed, short elevated
        signal_returns = np.concatenate([
            -elevated if len(elevated) > 0 else np.array([]),  # short when elevated
             depressed if len(depressed) > 0 else np.array([]),  # long when depressed
        ])
        sharpe = None
        if len(signal_returns) >= 4:
            m_sr = float(signal_returns.mean())
            s_sr = float(signal_returns.std())
            sharpe = round(m_sr / s_sr * np.sqrt(52), 3) if s_sr > 0 else None

        # Validated: either direction t-stat > 1.96 with sufficient signals
        elevated_sig  = (t_elev is not None and abs(t_elev) > 1.96 and len(elevated)  >= 5)
        depressed_sig = (t_dep  is not None and abs(t_dep)  > 1.96 and len(depressed) >= 5)
        validated = (elevated_sig or depressed_sig) and n_signals >= 8

        if validated:
            parts = []
            if elevated_sig:
                parts.append(
                    f"elevated z (n={len(elevated)}, t={t_elev:.2f}, p={p_elev:.3f}, "
                    f"mean 4W return {float(elevated.mean()) * 100:+.1f}%)"
                )
            if depressed_sig:
                parts.append(
                    f"depressed z (n={len(depressed)}, t={t_dep:.2f}, p={p_dep:.3f}, "
                    f"mean 4W return {float(depressed.mean()) * 100:+.1f}%)"
                )
            note = f"Sentiment signal statistically significant for: {'; '.join(parts)}."
        else:
            reasons = []
            if t_elev is not None:
                reasons.append(f"elevated t={t_elev:.2f} (p={p_elev:.3f})")
            if t_dep is not None:
                reasons.append(f"depressed t={t_dep:.2f} (p={p_dep:.3f})")
            note = (
                f"Sentiment signal not statistically significant (p < 0.05 required). "
                + ("; ".join(reasons) if reasons else "Insufficient data.")
            )

        out = {
            "validated":     validated,
            "n_signals":     n_signals,
            "n_elevated":    int(len(elevated)),
            "n_depressed":   int(len(depressed)),
            "elevated": {
                "mean_fwd_return_pct": round(float(elevated.mean()) * 100, 2) if len(elevated) else None,
                "t_stat":   round(t_elev, 3)  if t_elev  is not None else None,
                "p_value":  round(p_elev, 4)  if p_elev  is not None else None,
                "ci_95":    [round(ci_elev[0] * 100, 2), round(ci_elev[1] * 100, 2)]
                            if ci_elev[0] is not None else None,
                "significant": elevated_sig,
            },
            "depressed": {
                "mean_fwd_return_pct": round(float(depressed.mean()) * 100, 2) if len(depressed) else None,
                "t_stat":   round(t_dep, 3)  if t_dep  is not None else None,
                "p_value":  round(p_dep, 4)  if p_dep  is not None else None,
                "ci_95":    [round(ci_dep[0] * 100, 2), round(ci_dep[1] * 100, 2)]
                            if ci_dep[0] is not None else None,
                "significant": depressed_sig,
            },
            "signal_sharpe":   sharpe,
            "n_aligned_weeks": len(aligned),
            "confidence_note": note,
            "error":           None,
        }
        cache_set(ck, out)
        return out

    except Exception as e:
        logger.error(f"validate_sentiment_signal({ticker}): {e}", exc_info=True)
        return _val_error("sentiment", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 4. RUN FULL VALIDATION — Aggregate to HIGH/MEDIUM/LOW/UNVALIDATED
# ══════════════════════════════════════════════════════════════════════════════

def run_full_validation(ticker: str, sector: str = None) -> dict:
    """
    Run all three validation functions in parallel.
    Aggregate results to a single confidence tier and model card.
    """
    ck = cache_key(f"val_full_{ticker}_{sector or ''}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    tasks = {
        "factor":    (validate_factor_model,    (ticker, sector)),
        "earnings":  (validate_earnings_model,  (ticker,)),
        "sentiment": (validate_sentiment_signal, (ticker,)),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        future_map = {ex.submit(fn, *args): key for key, (fn, args) in tasks.items()}
        try:
            for fut in as_completed(future_map, timeout=180):
                key = future_map[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    results[key] = _val_error(key, str(e))
        except FuturesTimeout:
            for fut, key in future_map.items():
                if key not in results:
                    results[key] = _val_error(key, "timed out")

    factor_v   = results.get("factor",    {}).get("validated", False)
    earnings_v = results.get("earnings",  {}).get("validated", False)
    sentiment_v = results.get("sentiment", {}).get("validated", False)

    # Any hard data-gate errors (not just unvalidated)
    factor_err   = results.get("factor",    {}).get("error")
    earnings_err = results.get("earnings",  {}).get("error")
    sentiment_err = results.get("sentiment", {}).get("error")

    n_validated = sum([bool(factor_v), bool(earnings_v), bool(sentiment_v)])
    n_errored   = sum([bool(factor_err), bool(earnings_err), bool(sentiment_err)])

    if n_validated == 3:
        tier = "HIGH"
        tier_note = (
            "All three models pass out-of-sample validation. "
            "Predictions have demonstrated statistical reliability on historical data."
        )
    elif n_validated == 2:
        tier = "MEDIUM"
        tier_note = (
            "Two of three models pass out-of-sample validation. "
            "Treat outputs directionally; probability estimates carry moderate uncertainty."
        )
    elif n_validated == 1:
        tier = "LOW"
        tier_note = (
            "Only one model passes out-of-sample validation. "
            "Treat predictions as exploratory; do not rely on specific probability values."
        )
    else:
        if n_errored >= 2:
            tier = "UNVALIDATED"
            tier_note = (
                "Insufficient data to validate models — likely a short-history or low-coverage stock. "
                "Predictions shown are model outputs only, with no demonstrated out-of-sample accuracy."
            )
        else:
            tier = "UNVALIDATED"
            tier_note = (
                "No model passes out-of-sample validation for this stock. "
                "Predictions shown are model outputs only, with no demonstrated out-of-sample accuracy."
            )

    # Model-level card summaries
    def _card(key, res):
        if res.get("error"):
            return {"status": "DATA_GATE", "message": res["error"][:120], "validated": False}
        return {
            "status":    "PASS" if res.get("validated") else "FAIL",
            "validated": bool(res.get("validated")),
            "message":   res.get("confidence_note", "")[:200],
        }

    out = {
        "tier":          tier,
        "tier_note":     tier_note,
        "n_validated":   n_validated,
        "ticker":        ticker,
        "sector":        sector,
        "validated_at":  datetime.utcnow().isoformat() + "Z",
        "models": {
            "factor":    {**_card("factor",    results.get("factor",    {})),
                          **{k: v for k, v in results.get("factor", {}).items()
                             if k not in ("confidence_note", "error")}},
            "earnings":  {**_card("earnings",  results.get("earnings",  {})),
                          **{k: v for k, v in results.get("earnings", {}).items()
                             if k not in ("confidence_note", "error")}},
            "sentiment": {**_card("sentiment", results.get("sentiment", {})),
                          **{k: v for k, v in results.get("sentiment", {}).items()
                             if k not in ("confidence_note", "error")}},
        },
        "disclaimer": (
            "Statistical validation uses historical data only. Past predictive accuracy does "
            "not guarantee future performance. This tool is for research purposes and does not "
            "constitute investment advice."
        ),
        "error": None,
    }
    cache_set(ck, out)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCENARIO MODEL — Walk-forward backtest
# ══════════════════════════════════════════════════════════════════════════════

def validate_scenario_model(ticker: str) -> dict:
    """
    Walk-forward backtest of the DCF scenario model over the last 3 years.

    Methodology
    -----------
    For each quarterly checkpoint going back up to 12 quarters:
      1. Reconstruct TTM revenue and 2Y CAGR using only data available at
         that point (rolling slice of quarterly income statement history).
      2. Build simplified bear/base/bull scenario prices using the historical
         weekly price at that checkpoint as the starting point.
      3. Record the probability-weighted target (25/50/25 neutral weights).
      4. Compare to the actual price 52 weeks later.

    Because yfinance does not expose margin history, the DCF FCF computation
    is not fully replicated; instead, the price-range coverage test is based
    on ±1.5× the 2Y CAGR revenue shock translated through the current
    EV/Revenue multiple — a simplified but walk-forward-safe approach.

    Metrics
    -------
    mae_pct            : mean absolute error of prob-weighted target vs actual
    coverage_rate      : % of periods where actual fell inside bear–bull range
    narrative_dir_acc  : directional accuracy of narrative adjustment signal
                         (did stocks with high divergence underperform DCF base?)
    n_periods          : number of quarterly checkpoints tested
    validated          : True if coverage_rate ≥ 0.65 and n_periods ≥ 6
    """
    ck = cache_key(f"val_scenario_{ticker}")
    if (hit := cache_get(ck, CACHE_TTL)):
        return hit

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # ── Weekly price history (3 years) ────────────────────────────────────
        prices = _fetch_weekly(ticker, "3y")
        if prices.empty or len(prices) < 52:
            return _val_error("scenario", "Insufficient price history (< 52 weeks)")

        # ── Quarterly income statement ─────────────────────────────────────────
        try:
            qs = t.quarterly_income_stmt
        except Exception:
            qs = None
        if qs is None or qs.empty or "Total Revenue" in qs.index is False:
            return _val_error("scenario", "No quarterly revenue data available")

        rev_s = qs.loc["Total Revenue"].dropna().sort_index()
        if len(rev_s) < 6:
            return _val_error("scenario", "Fewer than 6 quarters of revenue data")

        # ── Current balance sheet inputs (static — limitation 1 in the doc) ───
        info       = t.info or {}
        shares     = float(info.get("sharesOutstanding") or 0)
        total_debt = float(info.get("totalDebt") or 0)
        total_cash = float(info.get("totalCash") or 0)
        net_cash   = total_cash - total_debt
        if shares <= 0:
            return _val_error("scenario", "Missing share count")

        gross_margin = float(info.get("grossMargins") or 0)
        tm_base = 18.0 if gross_margin > 0.60 else (14.0 if gross_margin > 0.40 else 10.0)

        try:
            tnx = yf.download("^TNX", period="5d", interval="1d",
                              progress=False, auto_adjust=True)
            col = (tnx["Close"] if "Close" in tnx.columns else tnx.iloc[:, 0]).dropna()
            rate_10y = float(col.iloc[-1])
        except Exception:
            rate_10y = 4.5
        beta = max(0.5, min(2.5, float(info.get("beta") or 1.0)))
        erp  = max(4.5, min(8.0, 4.5 + (beta - 1.0) * 1.5))
        dr   = (rate_10y + erp) / 100.0

        # ── Walk-forward loop: one checkpoint per quarter, up to 12 ──────────
        price_index   = prices.index
        price_values  = prices.values
        n_quarters    = min(len(rev_s) - 4, 12)   # need ≥4 quarters for TTM

        errors_pct   = []
        covered      = []

        for qi in range(n_quarters):
            # Data available at checkpoint qi (counting back from most recent)
            q_idx = len(rev_s) - 1 - qi       # most recent = 0 lag, then going back
            if q_idx < 3:
                break                          # need 4 quarters for TTM

            # TTM revenue at checkpoint
            ttm_at = float(rev_s.iloc[max(0, q_idx - 3): q_idx + 1].sum())
            if ttm_at <= 0:
                continue

            # 2Y CAGR at checkpoint (requires 8 quarters)
            if q_idx >= 7:
                ttm_2y = float(rev_s.iloc[q_idx - 7: q_idx - 3].sum())
                cagr   = (ttm_at / ttm_2y) ** 0.5 - 1.0 if ttm_2y > 0 else 0.08
            else:
                cagr = float(info.get("revenueGrowth") or 0) or 0.08

            # Approximate quarter date from rev_s index
            rev_dates = rev_s.index
            q_date    = rev_dates[q_idx] if hasattr(rev_dates[q_idx], "date") else None
            if q_date is None:
                continue

            # Find weekly price at or just after this quarter date
            start_idx = np.searchsorted(price_index, pd.Timestamp(q_date))
            if start_idx >= len(price_values):
                continue
            p_start = float(price_values[start_idx])
            if p_start <= 0:
                continue

            # Actual price 52 weeks later
            end_idx = start_idx + 52
            if end_idx >= len(price_values):
                continue
            p_actual = float(price_values[end_idx])

            # EV/Revenue at checkpoint
            ev_start  = p_start * shares + total_debt - total_cash
            ev_rev_at = ev_start / ttm_at if ttm_at > 0 else None
            if ev_rev_at is None or ev_rev_at <= 0:
                continue

            # Simplified scenario prices using current EV/Revenue multiple
            # Bear: 50% of CAGR, multiple −30%; Bull: 120% of CAGR, multiple +30%
            def _sp(g_mult, mult_adj):
                future_rev = ttm_at * (1 + cagr * g_mult)
                future_ev  = future_rev * ev_rev_at * mult_adj
                return max((future_ev + net_cash) / shares, 0.01)

            bp = _sp(0.50, 0.70)
            pp = _sp(1.00, 1.00)   # base
            up = _sp(1.20, 1.30)

            # Probability-weighted target (neutral 25/50/25)
            wt = 0.25 * bp + 0.50 * pp + 0.25 * up

            err_pct = abs(wt - p_actual) / p_actual * 100.0
            errors_pct.append(err_pct)
            covered.append(bp <= p_actual <= up)

        n_periods = len(errors_pct)
        if n_periods < 4:
            return _val_error("scenario",
                              f"Only {n_periods} valid walk-forward periods (need ≥ 4)")

        mae_pct       = round(float(np.mean(errors_pct)), 1)
        coverage_rate = round(float(np.mean(covered)), 3)
        validated     = coverage_rate >= 0.65 and n_periods >= 6

        out = {
            "validated":       validated,
            "n_periods":       n_periods,
            "mae_pct":         mae_pct,
            "coverage_rate":   coverage_rate,
            "confidence_note": (
                f"Bear–bull range contained actual price in "
                f"{round(coverage_rate * 100, 1)}% of {n_periods} walk-forward periods "
                f"(target ≥ 65%); MAE vs prob-weighted target: {mae_pct}%"
            ),
            "error": None,
        }
        cache_set(ck, out)
        return out

    except Exception as e:
        logger.error(f"validate_scenario_model({ticker}): {e}", exc_info=True)
        return _val_error("scenario", str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _val_error(model: str, msg: str) -> dict:
    return {
        "validated":       False,
        "confidence_note": f"Validation could not complete: {msg}",
        "error":           msg,
    }
