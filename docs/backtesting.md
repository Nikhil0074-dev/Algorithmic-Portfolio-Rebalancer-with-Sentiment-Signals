# Backtesting conventions
- Strategies: `buy_hold`, `periodic` (every N trading days, threshold 0), `threshold` (daily check), `sentiment` (threshold + bounded tilt).
- All strategies make the same initial purchase (fees included), so they differ only afterwards.
- Metrics: simple daily returns, 252-day year, volatility = std(ddof=1)*sqrt(252), Sharpe with rf=0 by default,
  turnover = traded value / average equity (two-way), max drawdown on the equity curve.
- Look-ahead tests (`tests/test_backtester.py`) prove that changing future prices or future sentiment cannot change past results.
- Do not tune thresholds on the same period you report. Split chronologically.
