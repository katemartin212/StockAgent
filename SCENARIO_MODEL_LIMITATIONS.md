# Scenario Analysis — Methodology Limitations

**File:** `predictive_analytics.py → _compute_dcf_core() + _apply_behavioral()`
**Last updated:** 2026-04-03
**Model version:** Three-stage DCF (replaced EV/Revenue percentile model)

This document records the known limitations of the scenario analysis calculation for future reference. It is intended as an honest accounting of where the model's outputs should be treated with caution, not as a reason to discard them — the model is directionally useful when its assumptions are understood.

---

## 1. Balance sheet inputs are point-in-time, applied to all of history

**What the model does:** EV/Revenue is computed for every historical weekly price using *current* values of shares outstanding, total debt, and total cash.

**The problem:** This creates a stale-inputs bias. Two years ago, the company likely had different debt, cash, and share counts. Applying today's balance sheet retroactively makes the historical EV/Rev series internally inconsistent — it measures something like "what would the EV/Rev have been two years ago if the capital structure had always looked like it does today."

**Practical impact:** For companies that have issued significant equity (dilution), bought back shares aggressively, raised debt, or paid down debt, the historical σ and z-score will be distorted. The direction of the distortion depends on the specific change: a company that has bought back 20% of shares since two years ago will show an artificially compressed historical EV/Rev, making the current level look less stretched than it actually is.

**What would fix it:** Fetching historical shares outstanding and balance sheet data at each point in time. yfinance does not expose weekly balance sheet history, so this would require a paid data provider.

---

## 2. Revenue is TTM (trailing), not forward

**What the model does:** All EV/Revenue calculations — both historical and in the scenario price projections — use TTM (trailing twelve month) revenue as the denominator.

**The problem:** For high-growth companies, TTM revenue systematically understates the revenue base the market is actually pricing. A company growing at 50% YoY already has a forward revenue run rate ~50% above TTM. Using TTM makes the current EV/Rev look more stretched than a forward EV/Rev would. Conversely, for declining companies, TTM overstates the forward revenue base.

**Practical impact:** The base scenario implied price (`revenue grows at trailing rate, multiple holds`) effectively double-counts growth — it applies a multiple that the market already set *expecting* future growth to a revenue figure that is then grown further. For high-growth companies this inflates the base case price significantly.

**What would fix it:** Using NTM (next twelve month) consensus revenue estimates as the denominator. This requires analyst estimate data (e.g., from a paid provider like Refinitiv or FactSet), which is not available via yfinance.

---

## 3. The log-normal assumption may not hold

**What the model does:** σ is computed on log-changes of the EV/Rev series, implicitly assuming EV/Rev is log-normally distributed.

**The problem:** Log-normality is a reasonable first-order assumption (multiples are bounded at zero, multiplicative, right-skewed) but empirical EV/Rev distributions have:
- **Fat tails:** extreme multiple compression events (bear markets, sector rotations) occur more frequently than a log-normal would predict.
- **Structural breaks:** multiples can shift regimes abruptly (e.g., a re-rating event). The log-normal model assumes a single stationary distribution.
- **Serial correlation:** multiple changes are not i.i.d. — they exhibit momentum (trending) and mean-reversion at longer horizons simultaneously.

**Practical impact:** The σ-based bear scenario underestimates tail risk in stress environments. In a severe de-rating (2000 dot-com, 2022 rate shock), actual multiple compression can reach 3–5σ events by the model's own measure.

---

## 4. σ is computed over the same 2-year window that contains the regime shift

**What the model does:** The MAD-based σ uses all 2 years of weekly EV/Rev log-changes.

**The problem:** If the stock underwent a significant re-rating during those 2 years (e.g., NVDA going from 5× to 20× EV/Rev), the σ of weekly log-changes will be inflated by the large directional moves during the transition. MAD reduces but does not eliminate this effect — a regime shift that played out gradually over many weeks still contributes many large log-changes that push up MAD.

**Consequence:** For a stock that has re-rated substantially, σ_annual will overstate the typical week-to-week multiple volatility in *stable* periods, producing wider bear and bull scenarios than the stock's "normal" behaviour warrants. The bear scenario may look implausibly pessimistic and the bull scenario implausibly optimistic when the multiple is actually quite stable at its new level.

**What would fix it:** Detecting structural breaks in the EV/Rev series (Chow test, CUSUM) and computing σ only on the post-break sub-period. This adds significant complexity and is sensitive to break-point detection errors.

---

## 5. Revenue growth multipliers are fixed and uncalibrated

**What the model does:** Bear = 50% of trailing growth, bull = 120% of trailing growth.

**The problem:** These multipliers are heuristic and identical for every company. A company whose revenue growth has historically been very stable (low volatility) gets the same ±50%/+20% stress as one whose growth swings wildly. In particular:
- For very high-growth companies (e.g., trailing growth 150%), the bear case of 50% growth (still 75% YoY) may not be bearish enough — the market scenario people actually worry about is a return to <20% growth.
- For mature, slow-growth companies (e.g., trailing growth 5%), the 50% haircut gives a 2.5% bear — which is barely a deceleration.

**What would fix it:** Using the historical standard deviation of YoY revenue growth rates to calibrate the bear/bull growth assumptions company-specifically. This data is available via yfinance quarterly financials but would require multi-year quarterly revenue history.

---

## 6. Net cash is applied identically across all three scenarios

**What the model does:** `implied_price = (future_EV + net_cash) / shares`, with the same net_cash in all three scenarios.

**The problem:** Net cash position will differ materially across scenarios. In the bear case, a high-growth company likely burns more cash (higher spending to try to sustain growth) or raises equity at depressed prices (dilution). In the bull case, it likely generates cash (improving unit economics). Applying a single static net_cash figure to all three scenarios overstates bear case prices (ignores cash burn / dilution) and understates bull case prices (ignores cash generation).

**Practical impact:** Most significant for pre-profitable or early-profitability companies. Less material for mature cash-generative businesses.

---

## 7. The z-knot calibration is heuristic

**What the model does:** The bear_k / bull_k asymmetry factors are linearly interpolated between four fixed z-knots: z ∈ {−1, 0, 1, 2} → bear_k ∈ {0.75, 1.0, 1.5, 2.0} and bull_k ∈ {1.5, 1.0, 0.75, 0.5}.

**The problem:** These values were chosen to be financially intuitive and internally consistent but are not derived from empirical data. There is no backtested evidence that, for example, a stock at log_z = 2 specifically warrants a 2.0σ bear scenario and 0.5σ bull scenario rather than 1.8σ / 0.6σ.

**Practical impact:** The absolute dollar values of bear and bull scenario prices are sensitive to these choices at extreme z values. The *direction* of the asymmetry (bear widens, bull narrows as z increases) is well-founded; the exact magnitudes are not.

---

## 8. σ is a 12-month horizon estimate using square-root-of-time scaling

**What the model does:** Weekly σ is scaled to annual by multiplying by √52, following the standard square-root-of-time rule.

**The problem:** Square-root-of-time scaling assumes i.i.d. returns (no serial correlation, no mean reversion). EV/Rev multiples are not i.i.d.:
- At short horizons, multiples exhibit momentum (serial correlation).
- At longer horizons (12+ months), they exhibit mean reversion.

The net effect is that for a 12-month scenario, the √52 scaling likely *overstates* the true uncertainty (because mean reversion pulls the multiple back), which makes both the bear and bull scenarios wider than they should be.

---

## 9. Probabilities assume stationarity of the historical distribution

**What the model does:** The bear/base/bull probabilities are derived from the percentile rank of the current EV/Rev within the 2-year historical distribution (ex last 13 weeks).

**The problem:** This assumes the historical distribution is a good guide to the probability of future outcomes. If the competitive landscape, interest rate environment, or company fundamentals have structurally changed, the historical distribution is not the relevant reference. A company that has permanently re-rated due to a business model shift (e.g., moving from hardware to software-as-a-service) should be benchmarked against its new peer group, not its own historical multiples.

---

## 10. Single-factor multiple model (EV/Revenue only)

**What the model does:** Uses only EV/Revenue as the valuation anchor.

**The problem:** EV/Revenue is an appropriate primary metric for high-growth, pre-profit companies but becomes less meaningful as companies mature. It ignores profitability, capital intensity, and cash conversion. Two companies with identical EV/Revenue multiples can have very different intrinsic value if one has 30% EBITDA margins and one has −10% EBITDA margins. The model does not distinguish between them.

**For IT sector coverage:** EV/Revenue is the right primary metric for software and early-growth tech. For semiconductors, infrastructure, and mature technology, EV/EBITDA or P/E would be more appropriate anchors. A future improvement would be to select the valuation metric based on profitability (use EV/Revenue if EBITDA margin < 15%, EV/EBITDA otherwise).

---

## 11. No macro or interest rate adjustment

**What the model does:** EV/Revenue multiples are taken as-is from market prices with no adjustment for the interest rate environment.

**The problem:** Growth stock EV/Revenue multiples are highly sensitive to real interest rates (the discount rate for long-duration cash flows). A stock that traded at 15× EV/Revenue in a ZIRP environment and at 5× in a 5% rate environment may both be "fairly valued" in their respective contexts. The historical σ and z-score mix observations from different rate regimes, making the distribution less stationary than it appears.

**Practical impact:** In periods of significant rate changes, the model will misinterpret rate-driven multiple compression as valuation normalisation and set the bear scenario too close to the current level.

---

## 12. yfinance data quality

- **Price data:** yfinance uses adjusted close prices (split and dividend adjusted), which is correct for return calculations but can create discontinuities around large special dividends.
- **Fundamental data:** Revenue, debt, and cash figures are sourced from Yahoo Finance's data pipeline. These occasionally contain errors or stale values, particularly for recently-reported quarters. No validation is performed against the raw SEC filings.
- **Survivorship bias:** The model only runs for tickers that currently exist and have 2 years of yfinance history. It cannot be backtested on companies that were acquired or went bankrupt.

---

## Summary table

| # | Limitation | Direction of bias | Severity |
|---|---|---|---|
| 1 | Balance sheet applied historically | Varies by company action | Medium |
| 2 | TTM revenue (not forward) | Overstates stretch for growth co's | High |
| 3 | Log-normal assumption | Understates tail risk | Medium |
| 4 | σ includes regime-shift period | Inflates σ, widens scenarios | Medium |
| 5 | Fixed revenue growth multipliers | Wrong direction for extreme growers | Medium |
| 6 | Static net cash across scenarios | Overstates bear price for pre-profit | Low–Medium |
| 7 | Heuristic z-knot calibration | Unknown; affects magnitude not direction | Low |
| 8 | √52 scaling assumes i.i.d. | Likely overstates 12-month σ | Low–Medium |
| 9 | Probability stationarity | Wrong for structurally re-rated co's | High |
| 10 | EV/Revenue only | Misleading for profitable/mature co's | High |
| 11 | No rate adjustment | Distorts cross-regime comparisons | Medium |
| 12 | yfinance data quality | Idiosyncratic, hard to predict | Low |
