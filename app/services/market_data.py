"""Market data providers behind one interface, with caching, retry and staleness checks."""
from __future__ import annotations

import time
import zlib
from abc import ABC, abstractmethod
from datetime import date
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class ProviderError(Exception):
    """Permanent provider failure (bad symbol, rate-limit message, malformed payload)."""


class TransientProviderError(ProviderError):
    """Temporary failure; safe to retry."""


def retry_with_backoff(func: Callable, retries: int = 3, base_delay: float = 0.5, factor: float = 2.0,
                       max_delay: float = 8.0, sleep: Callable[[float], None] = time.sleep):
    """Bounded exponential backoff. Only TransientProviderError is retried."""
    attempt = 0
    while True:
        try:
            return func()
        except TransientProviderError:
            if attempt >= retries:
                raise
            sleep(min(max_delay, base_delay * factor ** attempt))
            attempt += 1


class PriceProvider(ABC):
    name = "base"
    is_synthetic = False

    @abstractmethod
    def get_history(self, symbol: str, start, end) -> pd.DataFrame:
        """DataFrame indexed by tz-naive date with at least a 'close' column."""


_PROFILES: Dict[str, Tuple[float, float, float]] = {   # (annual drift, annual vol, start price)
    "AAPL": (0.12, 0.28, 130.0), "MSFT": (0.14, 0.26, 240.0), "NVDA": (0.30, 0.50, 15.0),
    "GOOGL": (0.12, 0.29, 90.0), "AMZN": (0.13, 0.32, 85.0), "TSLA": (0.15, 0.60, 110.0),
    "BND": (0.025, 0.06, 72.0), "SPY": (0.09, 0.17, 380.0),
}
_ANCHOR = pd.Timestamp("2022-01-03")
_N_DAYS = 4000


class SyntheticPriceProvider(PriceProvider):
    """Deterministic correlated GBM prices. For demos/tests only - NOT real market data."""
    name = "synthetic"
    is_synthetic = True

    def __init__(self, market_seed: int = 12345) -> None:
        self._market = np.random.default_rng(market_seed).standard_normal(_N_DAYS)
        self._cache: Dict[str, pd.Series] = {}

    def _series(self, symbol: str) -> pd.Series:
        if symbol not in self._cache:
            mu, vol, p0 = _PROFILES.get(symbol, (0.08, 0.30, 100.0 + zlib.crc32(symbol.encode()) % 100))
            idio = np.random.default_rng(zlib.crc32(symbol.encode())).standard_normal(_N_DAYS)
            beta = 0.0 if symbol == "BND" else 0.6
            z = beta * self._market + np.sqrt(1 - beta ** 2) * idio
            dt = 1 / 252
            rets = (mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
            idx = pd.bdate_range(_ANCHOR, periods=_N_DAYS)
            self._cache[symbol] = pd.Series(p0 * np.exp(np.cumsum(rets)), index=idx, name="close")
        return self._cache[symbol]

    def get_history(self, symbol: str, start, end) -> pd.DataFrame:
        if not symbol or not symbol.replace("-", "").replace(".", "").isalnum():
            raise ProviderError(f"Invalid symbol: {symbol!r}")
        s = self._series(symbol.upper())
        s = s.loc[pd.Timestamp(start).normalize(): pd.Timestamp(end).normalize()]
        df = s.round(4).to_frame("close")
        df.index.name = "date"
        return df


def parse_yahoo_chart(payload: dict) -> pd.DataFrame:
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ProviderError(f"Yahoo: {chart['error']}")
    try:
        res = chart["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("Yahoo: malformed response") from exc
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
    df = pd.DataFrame({"open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                       "close": q.get("close"), "volume": q.get("volume")}, index=idx)
    df.index.name = "date"
    return df.dropna(subset=["close"])


class YahooPriceProvider(PriceProvider):  # pragma: no cover - needs network
    """Uses Yahoo's unofficial chart endpoint. Review Yahoo's terms before any real use."""
    name = "yahoo"

    def get_history(self, symbol, start, end):
        import httpx

        def call():
            try:
                r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", timeout=20,
                              headers={"User-Agent": "Mozilla/5.0"},
                              params={"period1": int(pd.Timestamp(start).timestamp()),
                                      "period2": int(pd.Timestamp(end).timestamp()) + 86400, "interval": "1d"})
                if r.status_code in (429, 500, 502, 503, 504):
                    raise TransientProviderError(f"HTTP {r.status_code}")
                r.raise_for_status()
            except httpx.HTTPError as exc:
                raise TransientProviderError(str(exc)) from exc
            return r.json()

        return parse_yahoo_chart(retry_with_backoff(call))


def parse_alpha_vantage_daily(payload: dict) -> pd.DataFrame:
    for key in ("Error Message", "Information", "Note"):
        if key in payload:
            raise ProviderError(f"Alpha Vantage: {payload[key]}")
    series = payload.get("Time Series (Daily)")
    if not series:
        raise ProviderError("Alpha Vantage: no daily series in response")
    rows = {pd.Timestamp(d): {"open": float(v["1. open"]), "high": float(v["2. high"]), "low": float(v["3. low"]),
                              "close": float(v["4. close"]), "volume": float(v.get("5. volume", 0))}
            for d, v in series.items()}
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    return df


class AlphaVantagePriceProvider(PriceProvider):  # pragma: no cover - needs network + key
    name = "alphavantage"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is required")
        self.api_key = api_key

    def get_history(self, symbol, start, end):
        import httpx

        def call():
            try:
                r = httpx.get("https://www.alphavantage.co/query", timeout=20, params={
                    "function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full", "apikey": self.api_key})
                r.raise_for_status()
            except httpx.HTTPError as exc:
                raise TransientProviderError(str(exc)) from exc
            return r.json()

        df = parse_alpha_vantage_daily(retry_with_backoff(call))
        return df.loc[pd.Timestamp(start): pd.Timestamp(end)]


class CachedProvider(PriceProvider):
    """TTL cache wrapper to reduce API calls."""

    def __init__(self, inner: PriceProvider, ttl_seconds: float = 900, clock: Callable[[], float] = time.time) -> None:
        self.inner, self.ttl, self.clock = inner, ttl_seconds, clock
        self.name, self.is_synthetic = inner.name, inner.is_synthetic
        self._store: Dict[tuple, Tuple[float, pd.DataFrame]] = {}
        self.hits = self.misses = 0

    def get_history(self, symbol, start, end):
        key = (symbol, str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
        now = self.clock()
        hit = self._store.get(key)
        if hit and now - hit[0] < self.ttl:
            self.hits += 1
            return hit[1].copy()
        self.misses += 1
        df = self.inner.get_history(symbol, start, end)
        self._store[key] = (now, df)
        return df.copy()


def get_provider(name: str = "synthetic", api_key: str = "") -> PriceProvider:
    name = name.lower()
    if name == "synthetic":
        return SyntheticPriceProvider()
    if name == "yahoo":
        return CachedProvider(YahooPriceProvider())
    if name == "alphavantage":
        return CachedProvider(AlphaVantagePriceProvider(api_key))
    raise ValueError(f"Unknown price provider: {name}")
