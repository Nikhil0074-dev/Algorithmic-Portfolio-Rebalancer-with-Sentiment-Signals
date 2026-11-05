"""Portfolio valuation, target validation and drift analysis."""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd

CASH = "CASH"


def validate_targets(targets: Mapping[str, float], tol: float = 1e-6) -> None:
    """Target weights are fractions (0.25 = 25%) and must sum to 1 within tol."""
    if not targets:
        raise ValueError("Target allocation is empty")
    for k, v in targets.items():
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Target weight for {k} is not a finite number")
        if v < 0 or v > 1:
            raise ValueError(f"Target weight for {k} must be between 0 and 1")
    total = sum(targets.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Target weights must sum to 100% (got {total * 100:.4f}%)")


def position_values(quantities: Mapping[str, float], prices: Mapping[str, float]) -> Dict[str, float]:
    """Market value per non-cash position. Raises on missing/invalid price."""
    out: Dict[str, float] = {}
    for sym, qty in quantities.items():
        if sym == CASH:
            continue
        price = prices.get(sym)
        if price is None or not math.isfinite(price) or price <= 0:
            raise ValueError(f"Missing or invalid price for {sym}")
        if qty < 0:
            raise ValueError(f"Negative quantity for {sym}")
        out[sym] = qty * price
    return out


def portfolio_value(quantities: Mapping[str, float], prices: Mapping[str, float], cash: float) -> float:
    return sum(position_values(quantities, prices).values()) + cash


def drift_table(
    quantities: Mapping[str, float],
    prices: Mapping[str, float],
    cash: float,
    targets: Mapping[str, float],
    ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Current vs target allocation. Works with no sentiment (baseline)."""
    validate_targets(targets)
    values = position_values(quantities, prices)
    for a in targets:
        if a != CASH and a not in values:
            if a not in prices:
                raise ValueError(f"Missing or invalid price for {a}")
            values[a] = 0.0
    total = sum(values.values()) + cash
    if total <= 0:
        raise ValueError("Portfolio value must be positive")
    values[CASH] = cash
    ranges = ranges or {}
    rows = []
    for asset in sorted(values, key=lambda a: (a == CASH, a)):
        w = values[asset] / total
        tw = targets.get(asset, 0.0)
        lo, hi = ranges.get(asset, (tw, tw))
        rows.append({
            "asset": asset,
            "value": values[asset],
            "weight": w,
            "target_weight": tw,
            "drift_pp": (w - tw) * 100.0,
            "target_value": tw * total,
            "trade_value": tw * total - values[asset],  # >0 buy, <0 sell
            "in_range": (lo - 1e-12) <= w <= (hi + 1e-12),
        })
    return pd.DataFrame(rows)
