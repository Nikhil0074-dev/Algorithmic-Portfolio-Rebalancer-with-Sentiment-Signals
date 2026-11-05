"""Baseline + sentiment-aware rebalancing engine.

Design rules
  * Deterministic baseline first; sentiment is optional and bounded.
  * Sentiment can (a) flag trades for review or (b) tilt targets by at most
    +/- max_tilt (relative). It never replaces the user's targets, and it never
    triggers a trade by itself: the drift threshold is tested against the BASE target.
  * Risk constraints (cash, max weight, turnover, stale data) are applied after
    any tilt and are independent of sentiment.
Conventions
  * Weights are fractions. Trade quantity > 0 = buy, < 0 = sell.
  * Turnover = sum(|trade value|) / portfolio value (two-way, as in the spec).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set

import pandas as pd

from ..types import SentimentSignal
from .allocation import CASH, validate_targets

SENTIMENT_MODES = ("off", "review", "adjust")


@dataclass
class RebalanceConfig:
    threshold_pp: float = 3.0
    min_trade_value: float = 50.0
    fee_rate: float = 0.001
    fixed_fee: float = 0.0
    min_cash_weight: float = 0.0
    max_weight: Optional[float] = None
    fractional: bool = True
    max_turnover: Optional[float] = None
    restricted: frozenset = frozenset()
    sentiment_mode: str = "off"
    sentiment_k: float = 0.5
    max_tilt: float = 0.25
    sentiment_review_threshold: float = 0.4
    sentiment_min_confidence: float = 0.55
    sentiment_min_events: int = 2

    def __post_init__(self) -> None:
        if self.sentiment_mode not in SENTIMENT_MODES:
            raise ValueError(f"sentiment_mode must be one of {SENTIMENT_MODES}")
        if self.threshold_pp < 0 or self.min_trade_value < 0 or self.fee_rate < 0 or self.fixed_fee < 0:
            raise ValueError("threshold, min trade and fees must be non-negative")
        if not 0 <= self.min_cash_weight < 1:
            raise ValueError("min_cash_weight must be in [0, 1)")
        if self.max_weight is not None and not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.max_turnover is not None and self.max_turnover <= 0:
            raise ValueError("max_turnover must be positive")
        if not 0 <= self.max_tilt < 1:
            raise ValueError("max_tilt must be in [0, 1)")
        self.restricted = frozenset(self.restricted)


@dataclass
class RebalanceResult:
    trades: pd.DataFrame
    targets_used: Dict[str, float]
    portfolio_value: float
    cash_before: float
    cash_after: float
    turnover: float
    total_fees: float
    warnings: List[str] = field(default_factory=list)
    incomplete: bool = False

    def orders(self) -> Dict[str, float]:
        t = self.trades[self.trades["quantity"] != 0]
        return {r.symbol: float(r.quantity) for r in t.itertuples()}


# --------------------------------------------------------------------------- helpers
def _cap_weights(eff: Dict[str, float], max_w: float) -> Dict[str, float]:
    """Cap non-cash targets at max_w; the excess is moved to cash."""
    out = dict(eff)
    for a in list(out):
        if a != CASH and out[a] > max_w:
            out[CASH] = out.get(CASH, 0.0) + (out[a] - max_w)
            out[a] = max_w
    return out


def _enforce_min_cash(eff: Dict[str, float], min_w: float) -> Dict[str, float]:
    out = dict(eff)
    cash_w = out.get(CASH, 0.0)
    if cash_w >= min_w:
        return out
    noncash = sum(v for a, v in out.items() if a != CASH)
    if noncash <= 0:
        out[CASH] = 1.0
        return out
    scale = (1.0 - min_w) / noncash
    for a in out:
        if a != CASH:
            out[a] *= scale
    out[CASH] = min_w
    return out


def apply_sentiment_tilt(
    targets: Mapping[str, float],
    sentiment: Mapping[str, SentimentSignal],
    cfg: RebalanceConfig,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Experimental heuristic: w~ = w * clip(exp(k*S), 1-max_tilt, 1+max_tilt), renormalised
    so the total non-cash weight is unchanged. Ineligible signals get multiplier 1."""
    adj = dict(targets)
    noncash = [a for a, w in targets.items() if a != CASH and w > 0]
    mult: Dict[str, float] = {}
    for a in noncash:
        sig = sentiment.get(a)
        if sig is not None and sig.eligible(cfg.sentiment_min_confidence, cfg.sentiment_min_events):
            m = math.exp(cfg.sentiment_k * sig.score)
            mult[a] = min(1 + cfg.max_tilt, max(1 - cfg.max_tilt, m))
        else:
            mult[a] = 1.0
    tilted = {a: targets[a] * mult[a] for a in noncash}
    total_orig = sum(targets[a] for a in noncash)
    total_new = sum(tilted.values())
    if total_new > 0:
        for a in noncash:
            adj[a] = tilted[a] * total_orig / total_new
    effective = {a: (adj[a] / targets[a]) for a in noncash}
    return adj, effective


def _review(action: str, sig: Optional[SentimentSignal], cfg: RebalanceConfig) -> tuple[str, str]:
    """Return (signal_relation, priority)."""
    if cfg.sentiment_mode == "off" or sig is None or sig.score is None:
        return "n/a", "none"
    if not sig.eligible(cfg.sentiment_min_confidence, cfg.sentiment_min_events):
        return "uncertain", "low"
    if abs(sig.score) < cfg.sentiment_review_threshold:
        return "neutral", "low"
    if action == "HOLD":
        return "strong-no-trade", "medium"
    agrees = (action == "BUY" and sig.score > 0) or (action == "SELL" and sig.score < 0)
    return ("aligned", "medium") if agrees else ("conflict", "high")


# --------------------------------------------------------------------------- main
def rebalance(
    quantities: Mapping[str, float],
    prices: Mapping[str, float],
    cash: float,
    targets: Mapping[str, float],
    config: Optional[RebalanceConfig] = None,
    sentiment: Optional[Mapping[str, SentimentSignal]] = None,
    stale: Optional[Set[str]] = None,
) -> RebalanceResult:
    cfg = config or RebalanceConfig()
    validate_targets(targets)
    if cash < 0 or not math.isfinite(cash):
        raise ValueError("Cash must be a non-negative number")
    stale = set(stale or [])
    sentiment = dict(sentiment or {})
    warnings: List[str] = []
    incomplete = False

    held = {s: float(q) for s, q in quantities.items() if s != CASH}
    symbols = sorted(set(held) | {a for a in targets if a != CASH})
    for s in symbols:
        p = prices.get(s)
        if p is None or not math.isfinite(p) or p <= 0:
            raise ValueError(f"Missing or invalid price for {s}")
        if held.get(s, 0.0) < 0:
            raise ValueError(f"Negative quantity for {s}")
    values = {s: held.get(s, 0.0) * prices[s] for s in symbols}
    V = cash + sum(values.values())
    if V <= 0:
        raise ValueError("Portfolio value must be positive")

    base = {s: targets.get(s, 0.0) for s in symbols}
    base[CASH] = targets.get(CASH, 0.0)
    eff = dict(base)
    tilt_mult: Dict[str, float] = {}
    if cfg.sentiment_mode == "adjust":
        eff, tilt_mult = apply_sentiment_tilt(eff, sentiment, cfg)
    if cfg.max_weight is not None:
        capped = _cap_weights(eff, cfg.max_weight)
        if capped != eff:
            warnings.append(f"Target weights capped at {cfg.max_weight:.0%}; excess moved to cash")
        eff = capped
    eff = _enforce_min_cash(eff, cfg.min_cash_weight)

    # ---- 1. desired trades (quantities) with threshold / restrictions
    qty: Dict[str, float] = {}
    notes: Dict[str, List[str]] = {s: [] for s in symbols}
    hold_reason: Dict[str, str] = {}
    for s in symbols:
        p = prices[s]
        w = values[s] / V
        tw = eff[s]
        drift_pp = (w - tw) * 100.0
        delta = tw * V - values[s]
        force = cfg.max_weight is not None and w > cfg.max_weight + 1e-12
        reason = None
        if s in stale:
            reason = "price is stale - asset excluded"
            incomplete = True
            warnings.append(f"{s}: price data is stale; excluded from this proposal")
        elif s in cfg.restricted:
            reason = "restricted asset"
        elif not force and abs((w - base[s]) * 100.0) < cfg.threshold_pp - 1e-12:
            # Trigger is ALWAYS drift vs the user's base target, so sentiment can move the
            # destination of a trade but can never, by itself, cause trading.
            reason = f"drift {(w - base[s]) * 100.0:+.2f}pp vs base target is inside the +/-{cfg.threshold_pp:g}pp threshold"
        elif abs(delta) < 1e-6:
            reason = "already at target"
        elif abs(delta) < cfg.min_trade_value:
            reason = f"trade size ${abs(delta):,.2f} is below the ${cfg.min_trade_value:,.2f} minimum"
        q = 0.0 if reason else delta / p
        if q < 0:
            q = -min(-q, held.get(s, 0.0))
        if not cfg.fractional and q != 0:
            q = math.copysign(math.floor(abs(q) + 1e-9), q)
            if q == 0:
                reason = "rounds to zero whole shares"
        if force and q != 0:
            notes[s].append("position exceeds max weight - threshold overridden")
        qty[s] = q
        if reason:
            hold_reason[s] = reason

    def tv() -> float:
        return sum(abs(q) * prices[s] for s, q in qty.items())

    # ---- 2. turnover limit
    if cfg.max_turnover is not None and tv() / V > cfg.max_turnover + 1e-12:
        factor = cfg.max_turnover / (tv() / V)
        for s in qty:
            q = qty[s] * factor
            qty[s] = math.copysign(math.floor(abs(q) + 1e-9), q) if not cfg.fractional else q
            if qty[s] != 0:
                notes[s].append(f"scaled by {factor:.2f} for turnover limit")
        warnings.append(f"Trades scaled to respect turnover limit of {cfg.max_turnover:.0%}")

    # ---- 3. cash constraint (sells first, buys limited by available cash)
    def fee(value: float) -> float:
        return cfg.fee_rate * abs(value) + cfg.fixed_fee if value != 0 else 0.0

    min_cash = cfg.min_cash_weight * V
    proceeds = sum(-q * prices[s] - fee(q * prices[s]) for s, q in qty.items() if q < 0)
    available = cash + proceeds - min_cash

    def buy_cost() -> float:
        return sum(q * prices[s] + fee(q * prices[s]) for s, q in qty.items() if q > 0)

    guard = 0
    while buy_cost() > available + 1e-9 and guard < 100:
        guard += 1
        buys = [s for s, q in qty.items() if q > 0]
        if not buys:
            break
        if available <= 0:
            for s in buys:
                qty[s] = 0.0
                notes[s].append("buy dropped: not enough cash above reserve")
            break
        n = len(buys)
        var_cost = sum(qty[s] * prices[s] for s in buys) * (1 + cfg.fee_rate)
        scale = (available - n * cfg.fixed_fee) / var_cost if var_cost > 0 else 0.0
        if scale <= 0:
            smallest = min(buys, key=lambda x: qty[x] * prices[x])
            qty[smallest] = 0.0
            notes[smallest].append("buy dropped: not enough cash above reserve")
            continue
        scale = min(scale, 1.0) * (1 - 1e-12)
        for s in buys:
            q = qty[s] * scale
            qty[s] = math.floor(q + 1e-12) if not cfg.fractional else q
            notes[s].append("buy reduced to fit available cash")
        if not cfg.fractional and buy_cost() > available + 1e-9:
            smallest = min([s for s, q in qty.items() if q > 0], key=lambda x: qty[x] * prices[x])
            qty[smallest] = 0.0
    if any("cash" in " ".join(n) for n in notes.values()):
        warnings.append("Buys limited by available cash / minimum cash reserve")

    # ---- 4. drop buys that became too small after scaling (safe: only raises cash)
    for s in list(qty):
        if 0 < qty[s] * prices[s] < cfg.min_trade_value:
            qty[s] = 0.0
            hold_reason.setdefault(s, "buy below minimum trade size after cash limits")

    # ---- 5. build output
    total_fees = sum(fee(q * prices[s]) for s, q in qty.items())
    net_cash_flow = -sum(q * prices[s] for s, q in qty.items()) - total_fees
    cash_after = cash + net_cash_flow
    V_post = V - total_fees
    rows = []
    for s in symbols:
        q = qty[s]
        action = "BUY" if q > 0 else "SELL" if q < 0 else "HOLD"
        sig = sentiment.get(s)
        relation, priority = _review(action, sig, cfg)
        w = values[s] / V
        post_w = (held.get(s, 0.0) + q) * prices[s] / V_post
        parts = [f"{s} is {w:.1%} of the portfolio vs target {eff[s]:.1%} ({(w - eff[s]) * 100:+.1f}pp)."]
        if action == "HOLD":
            parts.append(f"No trade: {hold_reason.get(s, 'no change needed')}.")
        else:
            parts.append(f"{action.title()} {abs(q):,.4f} units (~${abs(q) * prices[s]:,.2f}).")
        if s in tilt_mult and abs(tilt_mult[s] - 1) > 1e-9:
            parts.append(f"Sentiment tilt moved target from {base[s]:.1%} to {eff[s]:.1%}.")
        if sig is not None and cfg.sentiment_mode != "off":
            if sig.score is None:
                parts.append("Sentiment unavailable.")
            else:
                parts.append(f"Sentiment {sig.score:+.2f} from {sig.n_events} events ({relation}).")
        parts.extend(n.capitalize() + "." for n in notes[s])
        rows.append({
            "symbol": s, "action": action, "quantity": q, "price": prices[s],
            "value": q * prices[s], "fee": fee(q * prices[s]),
            "current_weight": w, "base_target_weight": base[s], "target_weight": eff[s],
            "drift_pp": (w - eff[s]) * 100.0, "post_weight": post_w,
            "sentiment_score": (sig.score if sig is not None and cfg.sentiment_mode != "off" else None),
            "sentiment_relation": relation, "review_priority": priority,
            "rationale": " ".join(parts),
        })
    trades = pd.DataFrame(rows)
    return RebalanceResult(
        trades=trades, targets_used=eff, portfolio_value=V, cash_before=cash,
        cash_after=cash_after, turnover=tv() / V, total_fees=total_fees,
        warnings=warnings, incomplete=incomplete,
    )
