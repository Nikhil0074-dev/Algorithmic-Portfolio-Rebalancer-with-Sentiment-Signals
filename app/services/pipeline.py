"""Glue between providers, NLP and the engines (used by both the API and the dashboard)."""
from __future__ import annotations

import io
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from ..engine.allocation import CASH
from ..engine.risk import is_stale
from ..types import SentimentSignal
from .news import assign_events
from .sentiment import aggregate_sentiment, classify_articles
from .market_data import ProviderError


def last_business_day(today=None) -> pd.Timestamp:
    t = pd.Timestamp(today or pd.Timestamp.now(tz="UTC").tz_localize(None)).normalize()
    return t if t.weekday() < 5 else t - pd.offsets.BDay(1)


def latest_prices(provider, symbols: Iterable[str], as_of, max_age_bdays: int = 3, lookback_days: int = 20
                  ) -> Tuple[Dict[str, float], Dict[str, str], Set[str], Dict[str, str]]:
    """Returns (prices, price_dates, stale_symbols, errors). Missing data is reported, never invented."""
    as_of = pd.Timestamp(as_of).normalize()
    prices, dates, stale, errors = {}, {}, set(), {}
    for s in sorted(set(symbols)):
        if s == CASH:
            continue
        try:
            df = provider.get_history(s, as_of - pd.Timedelta(days=lookback_days), as_of)
            df = df[df.index <= as_of]
            if df.empty:
                raise ProviderError("no price rows returned")
            prices[s] = float(df["close"].iloc[-1])
            dates[s] = str(df.index[-1].date())
            if is_stale(df.index[-1], as_of, max_age_bdays):
                stale.add(s)
        except ProviderError as exc:
            errors[s] = str(exc)
    return prices, dates, stale, errors


def price_matrix(provider, symbols: Iterable[str], start, end) -> pd.DataFrame:
    cols = {}
    for s in sorted(set(symbols)):
        if s == CASH:
            continue
        cols[s] = provider.get_history(s, start, end)["close"]
    if not cols:
        raise ValueError("No tradable symbols")
    df = pd.concat(cols, axis=1).sort_index().ffill().dropna()
    return df


def load_news(news_provider, classifier, symbols: Iterable[str], start, end) -> pd.DataFrame:
    df = news_provider.get_news(list(symbols), start, end)
    df = assign_events(df)
    return classify_articles(df, classifier)


def current_signals(news_df: pd.DataFrame, symbols: Iterable[str], as_of_ts, **kw) -> Dict[str, SentimentSignal]:
    return {s: aggregate_sentiment(news_df, s, as_of_ts, **kw) for s in symbols if s != CASH}


def parse_holdings_csv(text: str, max_bytes: int = 200_000, max_rows: int = 1000) -> List[dict]:
    """symbol,asset_type,quantity,average_cost  (CASH row: quantity = amount)."""
    if len(text.encode()) > max_bytes:
        raise ValueError("CSV too large")
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"symbol", "quantity"}
    if not need.issubset(df.columns):
        raise ValueError("CSV must contain columns: symbol, quantity (optional: asset_type, average_cost)")
    if len(df) > max_rows:
        raise ValueError("Too many rows")
    rows, seen = [], set()
    for i, r in df.iterrows():
        sym = str(r["symbol"]).strip().upper()
        if not sym or sym == "NAN" or not sym.replace("-", "").replace(".", "").isalnum():
            raise ValueError(f"Row {i + 2}: invalid symbol")
        if sym in seen:
            raise ValueError(f"Row {i + 2}: duplicate symbol {sym}")
        seen.add(sym)
        try:
            q = float(r["quantity"])
        except (TypeError, ValueError):
            raise ValueError(f"Row {i + 2}: quantity is not a number") from None
        if not math.isfinite(q) or q < 0:
            raise ValueError(f"Row {i + 2}: quantity must be a finite non-negative number")
        try:
            cost = float(r["average_cost"]) if "average_cost" in df.columns and pd.notna(r.get("average_cost")) else 0.0
        except (TypeError, ValueError):
            raise ValueError(f"Row {i + 2}: average_cost is not a number") from None
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"Row {i + 2}: average_cost must be a finite non-negative number")
        atype = str(r["asset_type"]).strip().lower() if "asset_type" in df.columns and pd.notna(r.get("asset_type")) else ("cash" if sym == CASH else "stock")
        rows.append({"symbol": sym, "asset_type": atype, "quantity": q, "average_cost": cost})
    return rows
