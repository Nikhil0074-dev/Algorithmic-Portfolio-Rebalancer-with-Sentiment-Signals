import math

import numpy as np
import pandas as pd
import pytest

from app.engine import metrics as m
from app.engine.risk import concentration_flags, rolling_volatility, turnover, volatility_flag


def test_turnover():
    assert turnover([100, -300], 1000) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        turnover([1], 0)


def test_concentration_flags_ignore_cash():
    flags = concentration_flags({"AAPL": 0.45, "MSFT": 0.2, "CASH": 0.35}, 0.4)
    assert len(flags) == 1 and flags[0].startswith("AAPL")


def test_rolling_vol_and_flag():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.005, 200)
    wild = rng.normal(0, 0.05, 21)
    p = pd.Series(100 * np.cumprod(1 + np.concatenate([calm, wild])))
    assert volatility_flag(p)["flag"] is True
    assert volatility_flag(pd.Series(100 * np.cumprod(1 + calm)))["flag"] is False
    assert volatility_flag(p.head(10))["recent"] is None
    assert rolling_volatility(p).dropna().iloc[-1] > 0.3


EQ = pd.Series([100, 110, 99, 120.0], index=pd.bdate_range("2026-01-01", periods=4))


def test_metric_values():
    assert m.cumulative_return(EQ) == pytest.approx(0.2)
    assert m.max_drawdown(EQ) == pytest.approx(99 / 110 - 1)
    r = m.daily_returns(EQ)
    assert m.annualized_volatility(r) == pytest.approx(r.std(ddof=1) * math.sqrt(252))
    assert m.annualized_return(EQ) == pytest.approx(1.2 ** (252 / 3) - 1)
    assert m.sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))


def test_sharpe_zero_vol_is_nan_and_summary_cleans_it():
    flat = pd.Series([100.0] * 5, index=pd.bdate_range("2026-01-01", periods=5))
    assert math.isnan(m.sharpe_ratio(m.daily_returns(flat)))
    assert m.summarize(flat)["sharpe_ratio"] is None


def test_tracking_error_zero_vs_self_and_positive_otherwise():
    r = m.daily_returns(EQ)
    assert m.tracking_error(r, r) == pytest.approx(0)
    assert m.tracking_error(r, r * 0.5) > 0


def test_drawdown_series_nonpositive():
    assert (m.drawdown_series(EQ) <= 0).all()


def test_summary_turnover_and_costs():
    s = m.summarize(EQ, traded_value=EQ.mean(), fees=12.5, n_rebalances=3)
    assert s["turnover"] == pytest.approx(1.0) and s["transaction_costs"] == 12.5 and s["n_rebalances"] == 3
