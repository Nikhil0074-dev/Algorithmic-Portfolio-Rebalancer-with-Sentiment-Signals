"""News providers, de-duplication into events, and relevance."""
from __future__ import annotations

import difflib
import hashlib
import re
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import pandas as pd

from .market_data import ProviderError, retry_with_backoff, TransientProviderError

COLUMNS = ["article_id", "symbol", "title", "publisher", "url", "published_at", "available_at", "relevance"]
ALIASES = {
    "AAPL": ["apple", "aapl", "iphone"], "MSFT": ["microsoft", "msft", "azure"],
    "NVDA": ["nvidia", "nvda"], "GOOGL": ["alphabet", "google", "googl"],
    "AMZN": ["amazon", "amzn", "aws"], "TSLA": ["tesla", "tsla"],
}
COMPANY = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
           "AMZN": "Amazon", "TSLA": "Tesla", "BND": "Bond ETF", "SPY": "S&P 500 ETF"}


def empty_news() -> pd.DataFrame:
    df = pd.DataFrame(columns=COLUMNS)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    return df


# ---------------------------------------------------------------- normalisation
_SUFFIX = re.compile(r"\s+[-|–—]\s+[A-Za-z .&]+$")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "fbclid", "gclid"}


def normalize_title(title: str) -> str:
    t = _SUFFIX.sub("", (title or "").strip()).lower()
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", t)).strip()


def canonical_url(url: str) -> str:
    if not url:
        return ""
    p = urlparse(url.strip())
    q = [(k, v) for k, v in parse_qsl(p.query) if k.lower() not in _TRACKING]
    return urlunparse((p.scheme.lower(), p.netloc.lower().removeprefix("www."), p.path.rstrip("/"), "", urlencode(q), ""))


def relevance_score(title: str, symbol: str) -> float:
    """1.0 = company named, 0.3 = no explicit mention (kept low on purpose)."""
    t = (title or "").lower()
    names = ALIASES.get(symbol, [symbol.lower()])
    return 1.0 if any(re.search(rf"\b{re.escape(n)}\b", t) for n in names) else 0.3


def assign_events(df: pd.DataFrame, similarity: float = 0.85, window_hours: float = 48.0) -> pd.DataFrame:
    """Group near-identical stories (same symbol, close in time) into one event_id."""
    out = df.copy()
    if out.empty:
        out["event_id"] = pd.Series(dtype=str)
        return out
    out = out.sort_values("published_at", kind="stable").reset_index(drop=True)
    norm = [normalize_title(t) for t in out["title"]]
    urls = [canonical_url(u) for u in out["url"]]
    ev_ids: List[str] = []
    events: Dict[str, List[tuple]] = {}   # symbol -> [(event_id, norm, url, time)]
    for i, row in enumerate(out.itertuples()):
        cands = events.setdefault(row.symbol, [])
        cutoff = row.published_at - pd.Timedelta(hours=window_hours)
        cands[:] = [c for c in cands if c[3] >= cutoff]
        found = None
        for eid, n, u, _t in cands:
            if (urls[i] and u == urls[i]) or n == norm[i] or \
                    difflib.SequenceMatcher(None, n, norm[i]).ratio() >= similarity:
                found = eid
                break
        if found is None:
            found = "ev_" + hashlib.md5(f"{row.symbol}|{norm[i]}|{row.published_at.isoformat()}".encode()).hexdigest()[:10]
            cands.append((found, norm[i], urls[i], row.published_at))
        ev_ids.append(found)
    out["event_id"] = ev_ids
    return out


# ---------------------------------------------------------------- providers
class NewsProvider(ABC):
    name = "base"
    is_synthetic = False

    @abstractmethod
    def get_news(self, symbols: Iterable[str], start, end) -> pd.DataFrame: ...


_POS_T = [
    "{co} beats earnings estimates as demand surges ({t})",
    "{co} raises full-year guidance after record quarter ({t})",
    "Analysts upgrade {co} on strong growth outlook ({t})",
    "{co} wins major contract, shares climb ({t})",
]
_NEG_T = [
    "{co} misses revenue estimates, shares slump ({t})",
    "{co} faces antitrust probe over market practices ({t})",
    "Analysts downgrade {co} amid slowing demand ({t})",
    "{co} warns of supply delays and weak guidance ({t})",
]
_NEU_T = [
    "{co} to present at industry conference next week ({t})",
    "{co} announces date for quarterly results ({t})",
    "{co} names new regional head ({t})",
]
_TOPICS = ["cloud", "chips", "services", "devices", "advertising", "AI", "supply chain", "Europe", "Asia", "retail"]
_MARKET = ["Stocks rally as inflation cools", "Markets slump on rate concerns", "Wall Street ends flat in quiet trading",
           "Tech selloff deepens as yields climb", "Stocks climb on strong jobs data"]
_PUBS = ["Example Wire", "Sample Business Daily", "Demo Markets", "Placeholder Journal"]


class SyntheticNewsProvider(NewsProvider):
    """Deterministic fake headlines. Sentiment is INDEPENDENT of prices by construction,
    so it carries no predictive information - use it to test plumbing, not to judge alpha."""
    name = "synthetic-news"
    is_synthetic = True

    def __init__(self, seed: int = 7, articles_per_day: float = 0.6, duplicate_prob: float = 0.3) -> None:
        self.seed, self.apd, self.dup = seed, articles_per_day, duplicate_prob

    def get_news(self, symbols, start, end) -> pd.DataFrame:
        days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
        rows = []
        for sym in sorted(set(symbols)) + ["MARKET"]:
            seed = int(hashlib.md5(f"{self.seed}|{sym}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            co = COMPANY.get(sym, sym)
            for day in days:
                # draw everything per-day from a day-specific stream so results are range-independent
                drng = np.random.default_rng([seed, int(day.strftime("%Y%m%d"))])
                n = drng.poisson(self.apd if sym != "MARKET" else 0.4)
                for k in range(n):
                    hour, minute = int(drng.integers(12, 20)), int(drng.integers(0, 60))
                    ts = day.tz_localize("UTC") + pd.Timedelta(hours=hour, minutes=minute)
                    if sym == "MARKET":
                        title = str(drng.choice(_MARKET))
                    else:
                        pool = drng.choice(["p", "n", "u"], p=[0.3, 0.3, 0.4])
                        tmpl = drng.choice({"p": _POS_T, "n": _NEG_T, "u": _NEU_T}[str(pool)])
                        title = str(tmpl).format(co=co, t=str(drng.choice(_TOPICS)))
                    copies = 1 + int(drng.random() < self.dup)
                    for c in range(copies):
                        pub = _PUBS[int(drng.integers(0, len(_PUBS)))]
                        t2 = ts + pd.Timedelta(minutes=17 * c)
                        rows.append({
                            "article_id": hashlib.md5(f"{sym}|{title}|{t2}|{c}".encode()).hexdigest()[:12],
                            "symbol": sym,
                            "title": title if c == 0 else f"{title} - {pub}",
                            "publisher": pub,
                            "url": f"https://example.com/{sym.lower()}/{day:%Y%m%d}-{k}?utm_source=feed{c}",
                            "published_at": t2,
                            "available_at": t2 + pd.Timedelta(hours=1),
                            "relevance": 0.3 if sym == "MARKET" else 1.0,
                        })
        if not rows:
            return empty_news()
        df = pd.DataFrame(rows, columns=COLUMNS)
        return df.drop_duplicates("article_id").sort_values("published_at").reset_index(drop=True)


def parse_alpha_vantage_news(payload: dict, delay_hours: float = 1.0) -> pd.DataFrame:
    """Parse Alpha Vantage NEWS_SENTIMENT JSON. We ignore its sentiment fields and
    re-score titles with our own model so all sources are treated consistently."""
    for key in ("Error Message", "Information", "Note"):
        if key in payload:
            raise ProviderError(f"Alpha Vantage: {payload[key]}")
    rows = []
    for item in payload.get("feed", []):
        try:
            ts = pd.to_datetime(item["time_published"], format="%Y%m%dT%H%M%S", utc=True)
        except (KeyError, ValueError):
            continue  # missing / malformed timestamp -> excluded (cannot be used point-in-time)
        tickers = item.get("ticker_sentiment") or []
        for t in tickers:
            try:
                rel = float(t.get("relevance_score", 0.3))
            except (TypeError, ValueError):
                rel = 0.3
            rows.append({
                "article_id": hashlib.md5(f"{item.get('url')}|{t.get('ticker')}".encode()).hexdigest()[:12],
                "symbol": t.get("ticker"), "title": item.get("title", ""), "publisher": item.get("source", ""),
                "url": item.get("url", ""), "published_at": ts,
                "available_at": ts + pd.Timedelta(hours=delay_hours), "relevance": max(0.0, min(1.0, rel)),
            })
    return pd.DataFrame(rows, columns=COLUMNS) if rows else empty_news()


class AlphaVantageNewsProvider(NewsProvider):  # pragma: no cover - needs network + API key
    name = "alphavantage-news"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is required")
        self.api_key = api_key

    def get_news(self, symbols, start, end) -> pd.DataFrame:
        import httpx

        def call():
            try:
                r = httpx.get("https://www.alphavantage.co/query", timeout=20, params={
                    "function": "NEWS_SENTIMENT", "tickers": ",".join(symbols), "limit": 200,
                    "time_from": pd.Timestamp(start).strftime("%Y%m%dT0000"),
                    "time_to": pd.Timestamp(end).strftime("%Y%m%dT2359"), "apikey": self.api_key})
                r.raise_for_status()
            except httpx.HTTPError as exc:
                raise TransientProviderError(str(exc)) from exc
            return r.json()

        df = parse_alpha_vantage_news(retry_with_backoff(call))
        return df[df["symbol"].isin(set(symbols))].reset_index(drop=True)
