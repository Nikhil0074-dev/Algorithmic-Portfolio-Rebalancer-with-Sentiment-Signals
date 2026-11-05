# Architecture

```
dashboard.py (Streamlit)  ─┐
app/api.py (FastAPI)      ─┼─> app/services/pipeline.py ─> market_data.py (providers, cache, retry)
                           │                          └─> news.py (providers, dedupe) ─> sentiment.py (NLP, aggregation)
                           └─> app/engine/  allocation.py · rebalancer.py · risk.py · backtester.py · metrics.py
app/db.py (SQLite) stores portfolios, holdings, versioned targets, rebalance runs, proposed trades, backtests.
```
Key rules
1. Baseline rebalancing works with no sentiment; sentiment is optional (`off` / `review` / `adjust`).
2. The drift threshold is always tested against the user's **base** target, so sentiment can move a trade's destination
   (bounded by `max_tilt`) but cannot by itself trigger trading.
3. Risk constraints (cash reserve, max weight, turnover, stale data) apply after any tilt.
4. Backtest decisions at close *t* use data available at *t* and fill at close *t+1*; news is usable only after `available_at`.
5. Every run stores config, price snapshot and sentiment snapshot for audit.
