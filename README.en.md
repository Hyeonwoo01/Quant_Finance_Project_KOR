# Quant Finance Project KOR — English summary

A forecasting and validation toolkit for Korean equities (KOSPI·KOSDAQ stocks, ETFs and indices), built as a Streamlit app.
The main documentation is in Korean ([README.md](README.md), model notes in [MODELS.md](MODELS.md), experiment log in [EXPERIMENTS.md](EXPERIMENTS.md));
this page summarises what the project does, how it is validated, and what it found.

**The goal is not to prove a profitable strategy.** It is a tool for testing claims about predictability without look-ahead bias,
timing errors, ignored costs or overfitting, and for reporting the results honestly, including the negative ones.
It is a research and analysis tool, not a trading system, and nothing here is investment advice.

## What it does

| Tab | Content |
|---|---|
| Price range | t-distributed GARCH / GJR-GARCH simulation of 68%·95% price ranges and one-day VaR·ES. The one-day VaR blends GARCH variance with VKOSPI (implied volatility) for the KOSPI index and KODEX 200, and with an intraday high-low realized variance for other stocks |
| Direction | Probability that the price is higher in *h* trading days: price, macro, earnings, foreign/institutional flow and DART filing features; several models (logistic, gradient boosting, ensemble) chosen walk-forward, falling back to the historical up-rate when no model beats it |
| KOSPI ranking | Cross-sectional ranking of ~200 liquid KOSPI stocks (momentum, low volatility, flows, earnings, buybacks, value), including delisted stocks (KRX daily data), factor regressions, baseline strategies, market-impact capacity, cost sensitivity, rebalance-date sensitivity |
| Volatility · Regime | GARCH estimates vs realized volatility; a two-state Markov regime model and a "cash in the stressed regime" strategy |
| Forecast accuracy | Scoring of forecasts recorded in advance, GARCH backtests, Kupiec·Christoffersen VaR tests |

## How results are validated

- **Walk-forward only.** Every model is trained and selected with data available at the decision date. Tests change the data *after* a date and check that forecasts *up to* that date are unchanged.
- **Realistic timing and costs.** Decide at the close of day *t*, trade at the close of *t+1*. Commissions apply on both sides, and the Korean securities transaction tax applies on sells at the rate in force on the trade date. Price-limit (upper/lower limit) and trading-halt constraints are applied.
- **Honest statistics.** Overlapping-horizon t-statistics (Hansen–Hodrick / Newey–West), block-bootstrap confidence intervals for excess returns, and the Deflated Sharpe Ratio for selecting the best of many configurations. A strategy counts as having economic value only when the 95% lower bound of its after-cost excess return is above zero.
- **Pre-registration.** Before any model change is evaluated on real data, the candidate and its adoption rule are committed to [EXPERIMENTS.md](EXPERIMENTS.md); the change is adopted only if it passes that rule. Rejected candidates stay documented.
- **Forward record.** GitHub Actions records the day's forecasts for 35 tickers every weekday after the close and the weekly ranking picks every Friday. They are scored later. Records written after the next session has opened are not allowed.
- **Real-data report.** A weekly job re-runs the full validation on 35 tickers and commits [validation/README.md](validation/README.md).

## Key findings (real data, 35 tickers, Oct 2026)

- **Single-stock direction:** no evidence of predictive power (0 of 35 tickers beat the historical up-rate). Direction strategies and the regime strategy beat buy-and-hold after costs for 0 of 35.
- **KOSPI ranking:** it does rank stocks (IC ≈ 0.19, t ≈ 5–6), and the excess return was about +13–14% a year after costs and taxes. That return is explained by known factors (alpha ≤ 0) and a plain momentum baseline did about as well. Its size also varies a lot with the rebalance start date (+5% to +18%). In short, this is closer to factor investing than to new alpha.
- **Risk models:** volatility forecasts are useful. GARCH alone under-estimated KOSPI index risk (95% VaR exceeded on ~7% of days). Blending in VKOSPI fixed it (5.2%), and an intraday-range blend reduced clustered exceptions for individual stocks. With the adopted corrections, one-day VaR passes for 25 of 35 tickers and ES for 35 of 35. Several other candidates were tested and rejected under the same pre-registered rules.
- **Fixed while validating:** removing survivorship bias, missing-label handling and timing alignment cut an earlier ranking result from +20% to +12.9% a year.

## Run it

Python 3.12+ (tested on 3.12 and 3.13).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Optional API keys (DART, Bank of Korea ECOS, KRX Open API) go in `api_keys.env` (see `api_keys.env.example`). Without them the app runs on the remaining data.

Tests: `pip install -r requirements-dev.txt` then `pytest` (86 tests in 12 topic files, about 3–4 minutes; CI runs them on every push).

## Repository layout

| Path | What |
|---|---|
| `app.py`, `app_data.py`, `views/` | Streamlit app: shared settings and models, cached data loaders, one module per tab |
| `quant_engine.py` | GARCH / GJR-GARCH, VaR·ES backtests and tests, option pricing, statistics (HAC t, DSR, OLS) |
| `direction.py`, `regime.py`, `universe.py` | Direction models, regime model, KOSPI ranking (panel, walk-forward, evaluation, factors) |
| `market_data.py`, `alt_data.py`, `kr_data.py`, `kr_costs.py` | Data from Yahoo, Naver, DART, ECOS and KRX; Korean tax and costs |
| `forecast_log.py`, `validation.py`, `var_study.py` | Forward record, real-data validation report, VaR diagnostics |
| `tests/` | pytest suite (shared synthetic data in `conftest.py`) |
| `.github/workflows/` | CI tests, daily forecast record, weekly ranking and validation |

## Limitations

- Free data sources (Yahoo Finance, unofficial Naver endpoints, DART, ECOS, KRX) can change or lag without notice. Raw market data is not redistributed in this repository.
- Fills are assumed at the close. Order-book depth and closing-auction capacity are not modelled; market impact is approximated with a square-root rule in the ranking tab.
- The more assumptions are tried, the higher the chance of a lucky result. Conclusions rely on robustness checks and on the forward record.
