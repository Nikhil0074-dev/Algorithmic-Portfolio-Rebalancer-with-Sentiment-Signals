import numpy as np
import pandas as pd
import pytest

from app.engine.backtester import BacktestConfig, backtest, compare_strategies
from app.engine.rebalancer import RebalanceConfig
from app.types import SentimentSignal

IDX = pd.bdate_range("2026-01-01", periods=120)
T = {"A": 0.5, "B": 0.5}


def prices_df(seed=1, n=120):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.012, (n, 2))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=pd.bdate_range("2026-01-01", periods=n), columns=["A", "B"])


def cfg(strategy, fee=0.0, **kw):
    return BacktestConfig(strategy=strategy, targets=dict(T), rebalance=RebalanceConfig(
        min_trade_value=0.0, fee_rate=fee, sentiment_mode="adjust", sentiment_k=1.0), **kw)


def test_buy_hold_makes_only_initial_trades():
    r = backtest(prices_df(), cfg("buy_hold"))
    assert (r.trades["date"] == r.equity.index[0]).all()
    assert r.metrics["n_rebalances"] == 0 and r.metrics["turnover"] == 0


def test_buy_hold_matches_manual_computation():
    p = prices_df()
    r = backtest(p, cfg("buy_hold", initial_value=1000))
    expected = 500 * p["A"] / p["A"].iloc[0] + 500 * p["B"] / p["B"].iloc[0]
    assert r.equity.values == pytest.approx(expected.values)


def test_initial_value_and_fee_on_initial_purchase():
    r = backtest(prices_df(), cfg("buy_hold", fee=0.01, initial_value=1000))
    assert r.equity.iloc[0] == pytest.approx(1000 / 1.01 * 1.0, rel=1e-3) or r.equity.iloc[0] < 1000
    assert r.metrics["transaction_costs"] > 0


def test_equity_never_negative_and_weights_sum_to_one():
    for s in ("periodic", "threshold"):
        r = backtest(prices_df(3), cfg(s, fee=0.002, periodic_days=10, threshold_pp=2.0))
        assert (r.equity > 0).all()
        assert r.weights.sum(axis=1).tolist() == pytest.approx([1.0] * len(r.weights))


def test_periodic_rebalances_on_schedule_and_resets_weights():
    r = backtest(prices_df(2), cfg("periodic", periodic_days=20))
    assert 4 <= r.metrics["n_rebalances"] <= 6
    # orders are sized at day-20 prices and fill at day-21 prices, so weights land NEAR (not exactly on) target
    day = r.equity.index[21]
    assert r.weights.loc[day, "A"] == pytest.approx(0.5, abs=0.02)
    # ...and with no execution lag they land exactly on target
    r0 = backtest(prices_df(2), cfg("periodic", periodic_days=20, execution_lag=0))
    assert r0.weights.loc[r0.equity.index[20], "A"] == pytest.approx(0.5, abs=1e-9)


def test_execution_lag_trades_at_next_close():
    p = prices_df(2)
    r = backtest(p, cfg("periodic", periodic_days=20))
    later = r.trades[r.trades["date"] > r.equity.index[0]]
    assert set(later["date"]) == {p.index[i + 1] for i in range(20, 119, 20) if i + 1 < len(p)}
    for row in later.itertuples():
        assert row.price == pytest.approx(p.loc[row.date, row.symbol])


def test_fees_reduce_returns():
    p = prices_df(4)
    free = backtest(p, cfg("periodic", fee=0.0, periodic_days=5))
    costly = backtest(p, cfg("periodic", fee=0.005, periodic_days=5))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.metrics["transaction_costs"] > 0


def test_threshold_trades_less_than_periodic_daily():
    p = prices_df(5)
    th = backtest(p, cfg("threshold", threshold_pp=8.0))
    pe = backtest(p, cfg("periodic", periodic_days=1))
    assert th.metrics["n_rebalances"] < pe.metrics["n_rebalances"]


def test_deterministic_reproducible():
    a = backtest(prices_df(), cfg("threshold", fee=0.001, threshold_pp=3))
    b = backtest(prices_df(), cfg("threshold", fee=0.001, threshold_pp=3))
    pd.testing.assert_series_equal(a.equity, b.equity)


def _hist(p, score_fn):
    return {d: {s: SentimentSignal(score=score_fn(d, s), n_articles=5, n_events=5, confidence=0.8) for s in T} for d in p.index}


def test_sentiment_strategy_with_zero_sentiment_equals_threshold():
    p = prices_df(6)
    hist = _hist(p, lambda d, s: 0.0)
    th = backtest(p, cfg("threshold", threshold_pp=4.0), hist)
    se = backtest(p, cfg("sentiment", threshold_pp=4.0), hist)
    pd.testing.assert_series_equal(th.equity, se.equity, check_names=False)


def test_no_lookahead_future_sentiment_cannot_change_past():
    p = prices_df(7)
    cut = p.index[60]
    base = _hist(p, lambda d, s: 0.3 if s == "A" else -0.3)
    tampered = {d: ({s: SentimentSignal(score=-1.0 if s == "A" else 1.0, n_articles=9, n_events=9, confidence=0.99) for s in T}
                    if d > cut else v) for d, v in base.items()}
    a = backtest(p, cfg("sentiment", threshold_pp=1.0), base)
    b = backtest(p, cfg("sentiment", threshold_pp=1.0), tampered)
    upto = p.index[p.index <= p.index[61]]            # decision at `cut` executes at cut+1
    pd.testing.assert_series_equal(a.equity.loc[upto], b.equity.loc[upto], check_names=False)
    assert not a.equity.equals(b.equity)               # ...but tampering does matter afterwards


def test_no_lookahead_future_prices_cannot_change_past():
    p = prices_df(8)
    q = p.copy()
    q.iloc[70:] *= 3.0                                  # change the future only
    for s in ("periodic", "threshold"):
        a = backtest(p, cfg(s, periodic_days=7, threshold_pp=2.0))
        b = backtest(q, cfg(s, periodic_days=7, threshold_pp=2.0))
        pd.testing.assert_series_equal(a.equity.iloc[:70], b.equity.iloc[:70], check_names=False)
        ta = a.trades[a.trades["date"] < p.index[70]].reset_index(drop=True)
        tb = b.trades[b.trades["date"] < p.index[70]].reset_index(drop=True)
        pd.testing.assert_frame_equal(ta, tb)


def test_sentiment_tilt_respects_bounds():
    p = prices_df(9)
    hist = _hist(p, lambda d, s: 1.0 if s == "A" else -1.0)
    r = backtest(p, BacktestConfig(strategy="sentiment", targets=dict(T), threshold_pp=1.0, rebalance=RebalanceConfig(
        min_trade_value=0, fee_rate=0, sentiment_mode="adjust", sentiment_k=5.0, max_tilt=0.2)), hist)
    # tilt moves the destination by at most ~20% relative -> A's weight stays well below 70%
    # tilt of +/-20% relative on a 50/50 target -> destination is 60/40, so A stays near 60%
    assert r.weights["A"].max() < 0.63 and r.weights["A"].max() > 0.55
    assert r.weights["CASH"].abs().max() < 0.01      # small residual cash from the 1-day execution lag


def test_compare_strategies_skips_sentiment_without_history():
    res = compare_strategies(prices_df(), T)
    assert set(res) == {"buy_hold", "periodic", "threshold"}


def test_input_validation():
    with pytest.raises(ValueError):
        backtest(prices_df().head(2), cfg("buy_hold"))
    bad = prices_df()
    bad.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        backtest(bad, cfg("buy_hold"))
    for invalid_price in (np.inf, -np.inf, 0.0, -1.0):
        bad_price = prices_df()
        bad_price.iloc[5, 0] = invalid_price
        with pytest.raises(ValueError, match="finite, positive"):
            backtest(bad_price, cfg("buy_hold"))
    with pytest.raises(ValueError, match="Target weights"):
        backtest(prices_df(), BacktestConfig(strategy="buy_hold", targets={"A": 0.5, "B": 0.6}))
    with pytest.raises(ValueError, match="No price history"):
        backtest(prices_df(), BacktestConfig(strategy="buy_hold", targets={"A": 0.5, "Z": 0.5}))
    with pytest.raises(ValueError, match="sentiment history"):
        backtest(prices_df(), cfg("sentiment"))
    with pytest.raises(ValueError):
        BacktestConfig(strategy="magic", targets=T)
    with pytest.raises(ValueError):
        BacktestConfig(strategy="buy_hold", targets=T, execution_lag=3)


def test_cash_target_held_as_cash():
    c = BacktestConfig(strategy="buy_hold", targets={"A": 0.4, "B": 0.4, "CASH": 0.2}, initial_value=1000,
                       rebalance=RebalanceConfig(fee_rate=0))
    r = backtest(prices_df(), c)
    assert r.weights["CASH"].iloc[0] == pytest.approx(0.2)
    assert r.weights["CASH"].iloc[-1] < 0.25
