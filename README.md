# Algorithmic Portfolio Rebalancer with Sentiment Signals

Decision-support platform: portfolio drift analysis, financial-news sentiment, constrained and explainable rebalancing proposals,
and a look-ahead-safe backtester. **It never places orders and is not investment advice.**

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                                   # 148 tests
streamlit run dashboard.py               # dashboard on http://localhost:8501
uvicorn app.api:get_app --factory        # REST API, docs at http://localhost:8000/docs
```
Defaults use **synthetic** prices and news so everything runs offline. See `.env.example` to switch providers.

## What is implemented
| Module | Where |
|---|---|
| Portfolio valuation, target validation, drift table | `app/engine/allocation.py` |
| Baseline + sentiment-aware rebalancer (thresholds, fees, min trade, cash reserve, max weight, turnover cap, stale-data guard, whole shares, explanations) | `app/engine/rebalancer.py` |
| Risk checks (turnover, concentration, volatility flag, staleness) | `app/engine/risk.py` |
| Backtester (4 strategies, 1-day execution lag, fees) and metrics | `app/engine/backtester.py`, `metrics.py` |
| Providers (synthetic, Yahoo, Alpha Vantage) + cache + bounded retry | `app/services/market_data.py` |
| News providers, URL/title de-duplication into events | `app/services/news.py` |
| Sentiment classifier (lexicon default, optional FinBERT), time-decay aggregation, evaluation | `app/services/sentiment.py` |
| SQLite persistence with versioned targets and audit trail | `app/db.py` |
| REST API (15 endpoints) | `app/api.py` |
| Streamlit dashboard (6 pages) | `dashboard.py` |

## Sentiment modes
`off` baseline only · `review` flags trades whose direction agrees/conflicts with sentiment (quantities unchanged) ·
`adjust` tilts targets by `w*exp(k*S)` clipped to ±`max_tilt`, renormalised. Sentiment never triggers a trade on its own.

## Tests
`pytest` runs 148 tests: allocation, rebalancer constraints, sentiment maths, de-duplication, provider parsing/retry/cache,
risk and metrics, backtester (including look-ahead tests), DB, API (mocked provider) and every dashboard page (Streamlit AppTest).
Tests were also mutation-checked: deliberately breaking the cash reserve, threshold logic, news cut-off, drawdown and
execution lag each made tests fail.

## Screenshots
`screenshots/` holds real browser captures of each page (regenerate with `python scripts/capture_screenshots.py`).

