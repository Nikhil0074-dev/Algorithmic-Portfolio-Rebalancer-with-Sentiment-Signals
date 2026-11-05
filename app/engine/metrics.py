"""Performance metrics. Conventions: 252 trading days, simple daily returns."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

PERIODS = 252


def _clean(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cumulative_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(equity: pd.Series, periods: int = PERIODS) -> float:
    n = len(equity) - 1
    if n < 1 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (periods / n) - 1.0)


def annualized_volatility(returns: pd.Series, periods: int = PERIODS) -> float:
    return float(returns.std(ddof=1) * math.sqrt(periods)) if len(returns) > 1 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods: int = PERIODS) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - rf_annual / periods
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * math.sqrt(periods)) if sd > 1e-15 else float("nan")


def tracking_error(returns: pd.Series, bench: pd.Series, periods: int = PERIODS) -> float:
    a, b = returns.align(bench, join="inner")
    return float((a - b).std(ddof=1) * math.sqrt(periods)) if len(a) > 1 else float("nan")


def summarize(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    traded_value: float = 0.0,
    fees: float = 0.0,
    n_rebalances: int = 0,
) -> Dict[str, Optional[float]]:
    r = daily_returns(equity)
    avg_equity = float(equity.mean())
    out = {
        "cumulative_return": cumulative_return(equity),
        "annualized_return": annualized_return(equity),
        "annualized_volatility": annualized_volatility(r),
        "max_drawdown": max_drawdown(equity),
        "sharpe_ratio": sharpe_ratio(r, rf_annual),
        "turnover": traded_value / avg_equity if avg_equity > 0 else 0.0,
        "transaction_costs": float(fees),
        "n_rebalances": int(n_rebalances),
        "tracking_error": tracking_error(r, daily_returns(benchmark)) if benchmark is not None else float("nan"),
    }
    return {k: _clean(v) for k, v in out.items()}
