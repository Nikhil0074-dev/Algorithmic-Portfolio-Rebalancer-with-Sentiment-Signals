"""Risk checks that run independently of sentiment."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .allocation import CASH


def turnover(trade_values: Iterable[float], portfolio_value: float) -> float:
    """Two-way turnover: sum(|trade value|) / portfolio value."""
    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive")
    return float(sum(abs(v) for v in trade_values) / portfolio_value)


def concentration_flags(weights: Mapping[str, float], max_weight: float) -> List[str]:
    return [f"{a} is {w:.1%} (limit {max_weight:.0%})"
            for a, w in sorted(weights.items()) if a != CASH and w > max_weight + 1e-12]


def rolling_volatility(prices: pd.Series, window: int = 21, periods: int = 252) -> pd.Series:
    return prices.pct_change().rolling(window).std(ddof=1) * math.sqrt(periods)


def volatility_flag(prices: pd.Series, window: int = 21, ratio: float = 1.5) -> Dict[str, Optional[float]]:
    """Flag if recent vol is `ratio` times the longer-run vol of the same series."""
    rets = prices.pct_change().dropna()
    if len(rets) < window * 2:
        return {"recent": None, "long_run": None, "flag": False}
    recent = float(rets.tail(window).std(ddof=1) * math.sqrt(252))
    long_run = float(rets.std(ddof=1) * math.sqrt(252))
    return {"recent": recent, "long_run": long_run, "flag": bool(long_run > 0 and recent > ratio * long_run)}


def is_stale(last_date, as_of, max_age_bdays: int = 3) -> bool:
    """Business-day staleness check (weekends are not counted as staleness)."""
    last = np.datetime64(pd.Timestamp(last_date).date())
    now = np.datetime64(pd.Timestamp(as_of).date())
    if now < last:
        return False
    return int(np.busday_count(last, now)) > max_age_bdays


def data_freshness(last_dates: Mapping[str, object], as_of, max_age_bdays: int = 3) -> Dict[str, bool]:
    return {s: is_stale(d, as_of, max_age_bdays) for s, d in last_dates.items()}
