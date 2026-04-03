#!/usr/bin/env python3
"""
_ridge.py — Weighted Ridge regression with analytic standard errors.

Functions:
    weighted_ridge(X, y, weights)  GCV-tuned Ridge; returns (coefs, p_vals, r2, ci_lo, ci_hi)
"""

import numpy as np


def weighted_ridge(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray = None,
):
    """
    Weighted Ridge regression with cross-validated L2 penalty and analytic SEs.

    X must be pre-normalized (zero mean, unit std per column).
    weights are per-observation (unnormalized); None = uniform.
    Ridge alpha is selected via GCV (sklearn RidgeCV). SEs use the sandwich
    covariance  σ² · (X'WX + αI)⁻¹ · X'WX · (X'WX + αI)⁻¹.

    Returns (coefs, p_values, r_squared, ci_lo, ci_hi) or None on failure.
    ci_lo / ci_hi are 95% CI bounds on each slope coefficient.
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
        XtWX = Xc.T @ (w[:, None] * Xc)
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
