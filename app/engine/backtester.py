"""Event-driven daily backtester with look-ahead protection.

Timing model (execution_lag=1, the default):
  * Decision at the close of day t uses ONLY prices up to t and news available by t.
  * Orders are fixed in *quantities* at day-t prices and executed at the day t+1 close.
  * Fees are charged on every executed trade, including the initial purchase.
All strategies start from the same initial purchase so they differ only in later trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from ..types import SentimentSignal
from .allocation import CASH, validate_targets
from .metrics import summarize
from .rebalancer import RebalanceConfig, rebalance

STRATEGIES = ("buy_hold", "periodic", "threshold", "sentiment")


@dataclass
class BacktestConfig:
    strategy: str
    targets: Dict[str, float]
    initial_value: float = 100_000.0
    periodic_days: int = 21
    threshold_pp: float = 5.0
    execution_lag: int = 1
    rebalance: RebalanceConfig = field(default_factory=lambda: RebalanceConfig(
        min_trade_value=0.0, fee_rate=0.001))
    rf_annual: float = 0.0

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.execution_lag not in (0, 1):
            raise ValueError("execution_lag must be 0 or 1")
        if self.periodic_days < 1 or self.initial_value <= 0:
            raise ValueError("periodic_days and initial_value must be positive")


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, Optional[float]]


def _execute(orders: Mapping[str, float], prices: pd.Series, qty: Dict[str, float], cash: float,
             fee_rate: float) -> tuple[float, list]:
    """Sells first, then buys limited by cash. Returns new cash and executed trade records."""
    done = []
    for s, q in orders.items():
        if q < 0:
            q = -min(-q, qty.get(s, 0.0))
            if q == 0:
                continue
            val = -q * prices[s]
            fee = fee_rate * val
            qty[s] = qty.get(s, 0.0) + q
            cash += val - fee
            done.append((s, q, prices[s], q * prices[s], fee))
    buys = {s: q for s, q in orders.items() if q > 0}
    need = sum(q * prices[s] * (1 + fee_rate) for s, q in buys.items())
    scale = 1.0 if need <= cash + 1e-9 else max(0.0, cash / need) * (1 - 1e-12)
    for s, q in buys.items():
        q *= scale
        if q <= 0:
            continue
        val = q * prices[s]
        fee = fee_rate * val
        qty[s] = qty.get(s, 0.0) + q
        cash -= val + fee
        done.append((s, q, prices[s], val, fee))
    return max(cash, 0.0), done  # clamp float dust


def backtest(
    prices: pd.DataFrame,
    config: BacktestConfig,
    sentiment: Optional[Mapping[pd.Timestamp, Mapping[str, SentimentSignal]]] = None,
    benchmark: Optional[pd.Series] = None,
) -> BacktestResult:
    """prices: DataFrame (DatetimeIndex x symbols). sentiment: point-in-time signals by decision date."""
    if prices.empty or len(prices) < 3:
        raise ValueError("Need at least 3 price rows to backtest")
    if prices.isna().any().any():
        raise ValueError("Price matrix contains NaN; align/forward-fill before backtesting")
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("Price matrix must contain only finite, positive prices")
    prices = prices.sort_index()
    targets = config.targets
    validate_targets(targets)
    missing = [a for a in targets if a != CASH and a not in prices.columns]
    if missing:
        raise ValueError(f"No price history for: {missing}")
    if config.strategy == "sentiment" and sentiment is None:
        raise ValueError("The sentiment strategy requires a sentiment history")

    fee_rate = config.rebalance.fee_rate
    qty: Dict[str, float] = {}
    cash = config.initial_value
    p0 = prices.iloc[0]
    # initial purchase (identical for all strategies), net of fees
    budget = config.initial_value
    investable = {a: w for a, w in targets.items() if a != CASH}
    total_w = sum(investable.values()) + targets.get(CASH, 0.0)
    for a, w in investable.items():
        gross = budget * w / total_w / (1 + fee_rate)
        qty[a] = gross / p0[a]
    cash = max(0.0, budget - sum(q * p0[a] * (1 + fee_rate) for a, q in qty.items()))  # clamp float dust
    trades = [(prices.index[0], s, q, p0[s], q * p0[s], fee_rate * q * p0[s]) for s, q in qty.items()]

    base_cfg = replace(config.rebalance)
    pending: Optional[Dict[str, float]] = None
    pending_idx = -1
    equity, weights = [], []
    n_rebal = 0

    for i, (date, row) in enumerate(prices.iterrows()):
        if pending is not None and pending_idx == i:
            cash, done = _execute(pending, row, qty, cash, fee_rate)
            trades += [(date, *d) for d in done]
            n_rebal += 1 if done else 0
            pending = None
        vals = {s: qty.get(s, 0.0) * row[s] for s in qty}
        total = sum(vals.values()) + cash
        equity.append(total)
        weights.append({**{s: v / total for s, v in vals.items()}, CASH: cash / total})

        # --- decision for the close of this day (uses only data up to `date`)
        if i == 0 or i + config.execution_lag >= len(prices) or pending is not None:
            continue
        cfg = None
        sig = None
        if config.strategy == "periodic" and i % config.periodic_days == 0:
            cfg = replace(base_cfg, threshold_pp=0.0, sentiment_mode="off")
        elif config.strategy == "threshold":
            cfg = replace(base_cfg, threshold_pp=config.threshold_pp, sentiment_mode="off")
        elif config.strategy == "sentiment":
            cfg = replace(base_cfg, threshold_pp=config.threshold_pp,
                          sentiment_mode=base_cfg.sentiment_mode if base_cfg.sentiment_mode != "off" else "adjust")
            sig = sentiment.get(pd.Timestamp(date).normalize(), {})
        if cfg is None:
            continue
        res = rebalance(dict(qty), row.to_dict(), cash, targets, cfg, sentiment=sig)
        orders = res.orders()
        if orders:
            pending, pending_idx = orders, i + config.execution_lag
            if config.execution_lag == 0:
                cash, done = _execute(pending, row, qty, cash, fee_rate)
                trades += [(date, *d) for d in done]
                n_rebal += 1 if done else 0
                pending = None
                vals = {s: qty.get(s, 0.0) * row[s] for s in qty}
                total = sum(vals.values()) + cash
                equity[-1] = total
                weights[-1] = {**{s: v / total for s, v in vals.items()}, CASH: cash / total}

    eq = pd.Series(equity, index=prices.index, name=config.strategy)
    tdf = pd.DataFrame(trades, columns=["date", "symbol", "quantity", "price", "value", "fee"])
    body = tdf[tdf["date"] > prices.index[0]]
    metrics = summarize(eq, benchmark, config.rf_annual, traded_value=float(body["value"].abs().sum()),
                        fees=float(tdf["fee"].sum()), n_rebalances=n_rebal)
    return BacktestResult(config.strategy, eq, pd.DataFrame(weights, index=prices.index).fillna(0.0), tdf, metrics)


def compare_strategies(prices, targets, sentiment=None, benchmark=None, strategies=STRATEGIES, **kw) -> Dict[str, BacktestResult]:
    out = {}
    for s in strategies:
        if s == "sentiment" and sentiment is None:
            continue
        out[s] = backtest(prices, BacktestConfig(strategy=s, targets=dict(targets), **kw), sentiment, benchmark)
    return out
