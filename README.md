# CrossRank
### Walk-Forward Machine Learning for Cross-Sectional ETF Ranking

**A research project applying cross-sectional, walk-forward machine learning to rank UK-listed (London Stock Exchange) equity ETFs by expected relative performance, and constructing a long-only portfolio from the top-ranked names.**

---

## Table of Contents

1. [Motivation](#motivation)
2. [Research Question](#research-question)
3. [Key Terms (Glossary)](#key-terms-glossary)
4. [Methodology](#methodology)
   - [4.1 Universe Construction](#41-universe-construction)
   - [4.2 Equity-Like and Liquidity Filtering](#42-equity-like-and-liquidity-filtering)
   - [4.3 Target Definition](#43-target-definition)
   - [4.4 Feature Engineering](#44-feature-engineering)
   - [4.5 Walk-Forward Validation](#45-walk-forward-validation)
5. [Results](#results)
   - [5.1 Signal Discovery (V1 → V2 → V3)](#51-signal-discovery-v1--v2--v3)
   - [5.2 Model Comparison](#52-model-comparison)
   - [5.3 Feature Importance](#53-feature-importance)
   - [5.4 Portfolio Construction](#54-portfolio-construction)
   - [5.5 Transaction Costs](#55-transaction-costs)
   - [5.6 Regime Analysis](#56-regime-analysis)
   - [5.7 Robustness Testing](#57-robustness-testing)
6. [Comparison to the Original Study](#comparison-to-the-original-study)
7. [Limitations](#limitations)
8. [Repository Structure](#repository-structure)
9. [Reproducing the Results](#reproducing-the-results)

---

## Motivation

This project was inspired by a published study in which a data scientist built a walk-forward machine learning pipeline to rank approximately 6,800 US-listed exchange-traded funds (ETFs) by predicted future outperformance, across the period 2005–2023. The best-performing configuration (LightGBM, 1-month excess return target) reported a Sharpe ratio of 1.68 on a long/short, pre-transaction-cost backtest.

Rather than replicate his project on the same market and data, this project asks a different and arguably more interesting question: **does the same methodology generalise to a structurally different, much smaller market?** The UK ETF market is a genuine out-of-sample test — fewer funds, a shorter trading history for most of them, and a universe dominated by globally-diversified trackers rather than the sector/leveraged/thematic diversity of the US ETF landscape.

## Research Question

> Can a walk-forward machine learning model, using only point-in-time available information, rank UK-listed equity ETFs by their future relative performance well enough to construct a long-only portfolio that outperforms a passive benchmark — net of realistic trading costs, and consistently across different market regimes?

The project deliberately follows a staged, falsifiable research design: build the simplest possible model first (price-based features only), accept a null result if that is what the data shows, and add complexity (macro features, longer horizons, alternative models) only when there is a documented reason to.

## Key Terms (Glossary)

This section exists so the rest of the document is readable without a quant finance background. Skip if these are already familiar.

| Term | Meaning |
|---|---|
| **ETF** | Exchange-Traded Fund — a basket of securities (e.g. tracking an index) that trades on a stock exchange like a single share. |
| **Cross-sectional ranking** | Instead of forecasting a single fund's future return in isolation, the model ranks all eligible funds against each other in a given month — "which of these will do better than the others," not "what will this one return." |
| **Walk-forward validation** | A time-series-appropriate way of testing a model: train on a block of past months, test on the following few months, then roll the whole window forward and repeat. This avoids the model ever "seeing the future" during training, which a random train/test split would allow. |
| **Point-in-time / look-ahead bias** | A feature or piece of information is "point-in-time" if it was genuinely knowable at the moment a prediction was made. Look-ahead bias occurs when a model is accidentally given information from the future (even indirectly), which makes backtest results unrealistically good and untrustworthy. |
| **Information Coefficient (IC)** | The Spearman rank correlation between a model's predicted ranking of funds in a given month and what actually happened. An IC of 0 means no ranking skill (pure chance); an IC of 1 means a perfect ranking. Real-world financial ICs are typically small — 0.02–0.10 is considered a meaningful, usable signal. |
| **Newey-West standard errors** | A statistical adjustment used when data points are not fully independent of each other (e.g. overlapping time windows). It produces a more conservative, more honest significance test than a naive calculation would. |
| **Sharpe ratio** | A measure of risk-adjusted return: annualised return divided by annualised volatility. Higher is better; above 1.0 is generally considered strong for a systematic strategy. |
| **Basis point (bp)** | One hundredth of one percent (0.01%). Trading costs are typically quoted this way — "30bps" means 0.30%. |
| **UCITS ETF** | A European-regulated fund structure. Most UK-listed ETFs are UCITS funds domiciled in Ireland or Luxembourg (not the UK itself), even though they trade on the London Stock Exchange. |
| **Long-only vs. long/short** | A long-only portfolio only buys (holds) assets it expects to outperform. A long/short portfolio also actively bets against (shorts) assets expected to underperform, which mechanically produces larger swings in both return and risk. This project is long-only throughout. |
| **Turnover** | The proportion of a portfolio's holdings that change at each rebalance. High turnover means more trading, and therefore more trading costs. |
| **Equity-like filter** | A test used to confirm a fund genuinely behaves like an equity investment (as opposed to a bond, commodity, or money-market fund), based on its statistical relationship (beta and R²) to a broad equity benchmark. |
| **Beta** | How sensitive a fund's returns are to movements in a benchmark. A beta of 1.0 means it moves in line with the benchmark on average. |
| **R² (R-squared)** | The proportion of a fund's return variation that is explained by the benchmark. Ranges from 0 (no relationship) to 1 (fully explained). |
| **Ensemble** | A model built by combining the predictions of several other models, usually to improve robustness. |
| **LightGBM / Random Forest** | Two tree-based machine learning algorithms capable of learning non-linear relationships in data, commonly used in quantitative finance for exactly this kind of ranking problem. |

## Methodology

### 4.1 Universe Construction

The investable universe was built systematically, not by hand-selecting funds, because a cross-sectional ranking study is only as credible as the universe it ranks.

- **Source**: the `justetf-scraping` Python library, which queries justETF's structured fund database directly, filtered to `exchange='XLON'` (London Stock Exchange), `asset_class='class-equity'`, `instrument='ETF'`, and `strategy='epg-longOnly'` (excluding leveraged and inverse products at the source).
- This returned **1,597 candidate UK-listed equity ETFs**, of which **1,534** successfully resolved to usable price history via Yahoo Finance (a 97.5% hit rate).
- **Price history**: daily OHLCV data, 2008–2026, approximately 2.7 million rows.
- **Currency normalisation**: London-listed funds are quoted inconsistently in pence (GBX), pounds (GBP), or occasionally US dollars. All prices were normalised to a consistent GBP basis; where the exact quote currency could not be confirmed via API (121 of 1,534 tickers), a magnitude-based heuristic was applied and explicitly flagged in the metadata as inferred rather than confirmed.

**Documented limitation**: this universe reflects funds *currently* listed on justETF. It is not possible to fully reconstruct a true historical point-in-time universe (including funds that have since closed or delisted) using freely available data. This introduces a mild survivorship bias, consistent with the limitation most practitioners accept when working with free data sources, and is stated here rather than hidden.

### 4.2 Equity-Like and Liquidity Filtering

Not every fund tagged "equity" by a data provider is equally trustworthy for ranking purposes, and not every fund is liquid enough to be realistically tradeable. Two filters were applied, both computed **point-in-time** (using only information available up to the month in question, never later data):

**Equity-like filter.** Each fund's trailing 36-month beta and R² were computed against the MSCI ACWI (All Country World Index) ETF — a global equity benchmark, chosen specifically because the universe contains many globally-diversified funds (S&P 500, Nasdaq, MSCI World trackers) for which a UK-only benchmark like the FTSE 100 would be inappropriate (a Nasdaq tracker has little correlation to the FTSE 100 despite being unambiguously equity exposure).

The initial thresholds mirrored the original Oren Tapiero study (R² ≥ 0.50, beta ≥ 0.40, based on a 5-factor regression). Testing revealed these thresholds produced an *unstable* eligible universe over time — the pass rate swung from 61% (2020) down to 20% (2025) even while the total number of active funds grew steadily, indicating the instability was an artefact of market regime (the 2020–2021 period saw unusually synchronised global equity returns), not a real change in fund composition. After systematically testing a range of thresholds for both universe size and year-over-year stability, and spot-checking the funds included at the margin, the thresholds were recalibrated to **R² ≥ 0.25, beta ≥ 0.40** — large enough to retain a meaningful equity signal, stable enough not to be a regime artefact.

**Liquidity filter.** A 60-day trailing median traded value of at least £10,000/day, and no more than 30% zero-volume trading days in that window — removing effectively dead or untradeable funds from the eligible pool.

**Result**: the eligible universe grows from approximately **49 funds in 2011 to roughly 475 funds in 2025**, providing a genuinely large and statistically usable cross-section in the project's later years, while remaining honest about the much thinner universe in its earliest years.

### 4.3 Target Definition

The prediction target is each fund's **cross-sectionally demeaned return**: its own return in a given period, minus the equal-weighted average return of every other eligible fund in that same period.

This was a deliberate choice over using a fixed external benchmark (such as the FTSE 100 or a global index). Because the universe mixes UK-domestic funds with globally-diversified ones, no single external index is an appropriate yardstick for all of them simultaneously — a fixed benchmark would conflate genuine fund-selection skill with broader "UK vs. global" macro trends. Cross-sectional demeaning sidesteps this entirely: the benchmark each month is simply what the average eligible fund did that month, which is the mathematically correct target for a model whose entire purpose is to identify which funds will beat their peers.

### 4.4 Feature Engineering

**Price-based features (V1)**, computed from each fund's own daily price history:
- Trailing 1, 3, 6, and 12-month momentum (return)
- Realised volatility (trailing 60-day daily return standard deviation, annualised)
- Maximum drawdown over the trailing 12 months

**Macroeconomic features (V2)**, added on top of the price-based set:
- UK 10-year gilt yield (government bond yield — a proxy for interest rate conditions)
- SONIA (Sterling Overnight Index Average — the current reference short-term UK interest rate, replacing the discontinued LIBOR benchmark; data available from 2018 onward, with earlier months handled as missing values rather than dropped, since LightGBM and Random Forest can natively handle missing data)
- GBP/USD exchange rate
- VIX (the CBOE Volatility Index — a widely used global measure of market risk sentiment)

All macro and FRED-sourced series were obtained via the Federal Reserve Economic Data (FRED) database, which — despite being a US institution — hosts well-maintained UK series.

**Critical methodological point — avoiding look-ahead bias**: an initial version of this project contained a leakage bug in which the 1-month momentum feature was computed from data that overlapped almost entirely with the return the model was supposed to be predicting, producing an implausibly high and untrustworthy Information Coefficient (~0.72 — far beyond what any real financial signal would show). This was identified, diagnosed, and corrected by lagging every feature by one full period relative to its target, so that at any given prediction date, the model only ever sees information that was genuinely available *before* the period it is predicting. All results reported in this document reflect the corrected, leakage-free pipeline.

### 4.5 Walk-Forward Validation

Models were trained and evaluated using rolling walk-forward validation: a 36-month training window, followed by a 3-month out-of-sample test window, rolled forward one test-window at a time across the full dataset — never a random train/test split, which would be inappropriate for time-series data and would allow future information to leak into training.

## Results

### 5.1 Signal Discovery (V1 → V2 → V3)

The project followed a staged build, testing the simplest hypothesis first and only adding complexity where justified:

| Version | Features | Horizon | Mean IC | Statistical significance |
|---|---|---|---|---|
| V1 | Price-only (momentum, volatility, drawdown) | 1 month | −0.0168 | Not significant (p = 0.269) |
| V2 | Price + macro | 1 month | +0.0031 | Not significant (p = 0.867) |
| V3 | Price + macro | 3 months | **+0.0438** | **Significant (Newey-West p = 0.005)** |

**No statistically significant ranking signal was found at the 1-month horizon**, with either feature set. This is reported as a genuine finding, not a failure — a credible null result is more valuable than a forced positive one. A real, statistically robust signal emerged only once the prediction horizon was extended to 3 months, consistent with the intuition that monthly fund returns are dominated by short-term noise that washes out over a slightly longer window.

### 5.2 Model Comparison

At the 3-month horizon, four model types were tested, following the original study's design (Linear Regression, Random Forest, LightGBM, and an ensemble):

| Model | Mean IC | Newey-West p-value |
|---|---|---|
| Linear Regression | **−0.0468** | 0.043 (significantly *negative*) |
| LightGBM | +0.0377 | 0.016 |
| Random Forest | +0.0425 | 0.010 |
| **Ensemble (RF + LightGBM, rank-averaged)** | **+0.0438** | **0.005** |
| Ensemble (RF + LightGBM + LR) | +0.0187 | 0.318 (not significant) |

Two findings stand out. First, **the signal is non-linear**: Linear Regression does not merely fail to find it, it actively ranks funds in the wrong direction, confirming that tree-based models are genuinely required here, not just marginally better. Second, **a disciplined two-model ensemble outperforms every individual model**, while including the under-performing Linear Regression model in a three-way ensemble *drags performance back down* — direct evidence that blending a weak or wrong model is actively harmful, not merely neutral.

### 5.3 Feature Importance

The single most important feature for both LightGBM and Random Forest is the **UK 10-year gilt yield**, ahead of every price-based momentum or volatility feature. This echoes a core finding from the original study, in which macroeconomic variables (growth score, money supply growth, market stress score) dominated feature importance ahead of price momentum — suggesting that, in both studies, the model is partly learning to time the macro regime, not purely picking individual winning funds.

### 5.4 Portfolio Construction

Using the validated 3-month ensemble signal, several long-only portfolio construction methods were tested on a quarterly rebalancing schedule (matching the 3-month prediction horizon exactly, avoiding any ambiguity about overlapping holding periods):

| Portfolio | Gross Annualised Return | Gross Sharpe |
|---|---|---|
| **Top 10% equal-weighted** | **17.2%** | **1.20** |
| Top 20% equal-weighted | 14.0% | 1.11 |
| Top 10 funds (fixed count) | 20.1% | 1.01 |
| Top 20 funds (fixed count) | 17.7% | 1.07 |
| Score-weighted (top 20%) | 14.5% | 1.12 |
| *Passive benchmark (equal-weight, all eligible funds)* | *8.7%* | *0.79* |

The **Top 10% equal-weighted** construction was found to be the best-performing and most robust choice, beating the passive benchmark on both return and risk-adjusted terms, and comfortably ahead of a simple FTSE 100 tracker (dividend-adjusted) over the same period. Over the full 2013–2026 out-of-sample period, £1 invested in this strategy grew to **£8.22**, versus £3.00 for the passive eligible-universe benchmark and £1.65 for a FTSE 100 tracker.

Portfolio size was further tested across a wider range (Top 5%, 10%, 15%, 20%, 30%); Sharpe ratio peaks cleanly and smoothly at Top 10% rather than at an isolated, potentially coincidental point — see [Section 5.7](#57-robustness-testing).

### 5.5 Transaction Costs

A strategy with a genuine but modest statistical edge is exactly the kind of result that can be eliminated entirely by realistic trading costs — so this was tested explicitly, rather than only reporting an idealised gross return.

Average portfolio turnover was approximately **78% per quarterly rebalance** (roughly 300%+ annualised). Three cost scenarios were tested, reflecting the realistic range of bid/ask spreads and brokerage costs across UK-listed ETFs of varying liquidity (note: most UK-listed ETFs are UCITS funds domiciled in Ireland or Luxembourg, and are therefore exempt from UK Stamp Duty Reserve Tax, which does not apply here):

| Cost scenario | Round-trip cost | Net Sharpe |
|---|---|---|
| Gross (no costs) | — | 1.20 |
| Low | 15 basis points | 1.17 |
| Base case | 30 basis points | 1.13 |
| High | 60 basis points | 1.06 |

The strategy's net Sharpe ratio remains above 1.0 in every cost scenario tested, comfortably clearing the passive benchmark's gross Sharpe of 0.79 even under the most pessimistic cost assumption.

### 5.6 Regime Analysis

Performance was broken out across different market environments to check whether the strategy's edge is concentrated in a specific regime (which would be an important caveat) or genuinely broad-based:

The strategy outperformed the passive benchmark in **every regime tested**: high and low volatility periods (classified by VIX relative to its own historical median), rising and falling UK interest rate environments (classified by the 6-month change in the 10-year gilt yield), and — directionally, though with too small a sample to draw firm statistical conclusions (n = 3 quarters) — during the 2020 COVID period. No regime was found in which the edge disappeared or reversed.

### 5.7 Robustness Testing

Two core modelling assumptions were stress-tested to check the result was not fragile to arbitrary design choices:

**Training window length.** The signal was re-tested using 24-month, 36-month, and 60-month training windows (instead of only the original 36-month choice):

| Training window | Mean IC | Newey-West p-value |
|---|---|---|
| 24 months | 0.0458 | 0.024 |
| **36 months (primary)** | **0.0499** | **0.001** |
| 60 months | 0.0380 | 0.026 |

The signal remains statistically significant across all three window lengths, with the original 36-month choice performing strongest — reassuring evidence this was a reasonable design decision rather than a lucky, overfit pick.

**Portfolio size.** As shown in Section 5.4, Sharpe ratio was tested across Top 5%, 10%, 15%, 20%, and 30% portfolios:

Sharpe peaks cleanly and smoothly at the Top 10% level (1.20), with a gradual decline on either side (1.11 at Top 5%, 1.08 at Top 30%) — evidence this is a genuine optimum, not an isolated spike that happened to look good in isolation.

## Comparison to the Original Study

| Aspect | Original study | CrossRank (this project) |
|---|---|---|
| Market | United States | United Kingdom |
| Universe size | ~6,800 funds | ~49 (2011) growing to ~475 (2025) |
| Universe construction | Not fully disclosed | Fully systematic (justETF API filters), documented |
| Equity-like filter | FF5 R² ≥ 0.50, beta ≥ 0.40 | ACWI R² ≥ 0.25, beta ≥ 0.40 (recalibrated for regime stability) |
| Portfolio type | Long/short | Long-only |
| Best reported Sharpe | 1.68 (pre-transaction-cost) | 1.20 gross / 1.06–1.13 net of costs |
| Transaction costs | Not tested (explicitly pre-cost) | Tested across three realistic cost scenarios |
| Regime robustness | Not reported | Tested explicitly, edge holds across all tested regimes |
| Significance testing | Not disclosed | Newey-West HAC-adjusted, accounting for overlapping windows |

The two studies are not directly comparable in magnitude, for several defensible reasons: Oren's much larger universe affords greater statistical power and more stable decile-based portfolios; his universe construction may not have fully excluded leveraged or inverse products, which carry mechanical return patterns distinct from genuine stock-picking skill; a long/short construction is mechanically larger in both return and volatility than a long-only one; and as a single published post, there is no visibility into how many alternative configurations may have been tested before the reported one was selected. This project's smaller, more modest, but independently and transparently validated result is considered a legitimate and arguably more rigorous finding, not a lesser one.

## Limitations

- **Survivorship bias**: the universe reflects funds currently listed via justETF's database; funds that closed or delisted prior to the data snapshot are not represented in the historical record. This is a known limitation of working with freely available data and could not be fully resolved.
- **Monthly overlapping rebalancing**: an attempt was made to test a monthly (rather than quarterly) rebalancing schedule with overlapping 3-month holding periods. This was found to require more careful tranche-level return accounting than initially implemented, produced an implausible result, and was abandoned rather than reported — the primary results in this document use quarterly, non-overlapping rebalancing throughout.
- **Overlapping prediction windows**: because the target is a 3-month forward return, adjacent monthly observations share two of three months of underlying return data, introducing serial correlation. This was explicitly addressed using Newey-West standard errors rather than naive statistical tests throughout.
- **COVID-period sample size**: the regime analysis's COVID-period finding (n = 3 quarters) is directionally consistent with the strategy's broader strength in high-volatility regimes but is too small a sample to treat as independently conclusive.
- **No single-stock data**: this project operates purely on a cross-section of funds, not underlying equities, consistent with the original study's ETF-level design.

## Repository Structure

```
CrossRank/
├── src/                          # Reusable, checkpointed pipeline code
│   ├── config.py                 # Central configuration (parameters, ticker lists)
│   ├── framework.py              # Step/CheckpointManager base classes
│   └── universe.py               # Universe audit and data collection logic
├── notebooks/
│   ├── 01_universe_data.ipynb        # Universe sourcing, price data collection, cleaning
│   ├── 02_universe_filtering.ipynb   # Equity-like and liquidity filtering
│   ├── 03_features_targets.ipynb     # Feature engineering, target construction
│   ├── 04_walk_forward_model.ipynb   # Walk-forward model training and comparison
│   ├── 05_portfolio_construction.ipynb  # Portfolio construction, costs, regime, robustness
│   └── 06_final_report.ipynb         # Chart generation for this report
├── results/
│   ├── charts/                   # All figures referenced in this document
│   └── tables/                   # Underlying result tables (CSV/JSON)
└── README.md                     # This document
```

## Reproducing the Results

This project uses free, publicly accessible data sources throughout:
- **ETF universe and metadata**: [justETF](https://www.justetf.com) via the open-source `justetf-scraping` library
- **Price history**: Yahoo Finance via `yfinance`
- **Macroeconomic data**: Federal Reserve Economic Data (FRED) via `pandas-datareader`

Notebooks are numbered and designed to be run in sequence; each stage's output is checkpointed, so the full pipeline can be paused and resumed without recomputation. Note that the full price-history dataset (~220MB) is not included in this repository due to GitHub file size limits, and can be regenerated by running `01_universe_data.ipynb`.

Core dependencies: `pandas`, `numpy`, `yfinance`, `lightgbm`, `scikit-learn`, `scipy`, `statsmodels`, `pandas-datareader`, `justetf-scraping`.
