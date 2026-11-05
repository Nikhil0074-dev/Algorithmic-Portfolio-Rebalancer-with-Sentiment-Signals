import pandas as pd
import pytest

from app.engine.risk import is_stale
from app.services.market_data import (CachedProvider, ProviderError, SyntheticPriceProvider, TransientProviderError,
                                      get_provider, parse_alpha_vantage_daily, parse_yahoo_chart, retry_with_backoff)


def test_synthetic_deterministic_and_consistent_across_ranges():
    a, b = SyntheticPriceProvider(), SyntheticPriceProvider()
    x = a.get_history("AAPL", "2026-01-01", "2026-10-30")
    y = b.get_history("AAPL", "2026-01-01", "2026-10-30")
    pd.testing.assert_frame_equal(x, y)
    z = a.get_history("AAPL", "2026-06-01", "2026-10-30")
    assert z["close"].iloc[-1] == x["close"].iloc[-1]
    assert (x["close"] > 0).all() and x.index.is_monotonic_increasing and x.index.tz is None


def test_synthetic_symbols_differ_and_invalid_rejected():
    p = SyntheticPriceProvider()
    assert p.get_history("AAPL", "2026-01-01", "2026-02-01")["close"].iloc[-1] != p.get_history("MSFT", "2026-01-01", "2026-02-01")["close"].iloc[-1]
    with pytest.raises(ProviderError):
        p.get_history("A B;DROP", "2026-01-01", "2026-02-01")


def test_retry_succeeds_after_transient_failures_with_bounded_backoff():
    calls, sleeps = {"n": 0}, []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientProviderError("boom")
        return "ok"

    assert retry_with_backoff(flaky, retries=3, base_delay=1, factor=2, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]


def test_retry_gives_up_after_limit():
    sleeps = []
    with pytest.raises(TransientProviderError):
        retry_with_backoff(lambda: (_ for _ in ()).throw(TransientProviderError("x")), retries=2, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_retry_caps_delay():
    sleeps = []
    with pytest.raises(TransientProviderError):
        retry_with_backoff(lambda: (_ for _ in ()).throw(TransientProviderError("x")), retries=6, base_delay=1, factor=10,
                           max_delay=5, sleep=sleeps.append)
    assert max(sleeps) == 5


def test_permanent_errors_not_retried():
    sleeps, n = [], {"c": 0}

    def bad():
        n["c"] += 1
        raise ProviderError("nope")

    with pytest.raises(ProviderError):
        retry_with_backoff(bad, sleep=sleeps.append)
    assert n["c"] == 1 and sleeps == []


def test_cache_hit_and_ttl_expiry():
    now = {"t": 0.0}
    inner = SyntheticPriceProvider()
    cp = CachedProvider(inner, ttl_seconds=60, clock=lambda: now["t"])
    cp.get_history("AAPL", "2026-01-01", "2026-02-01")
    cp.get_history("AAPL", "2026-01-01", "2026-02-01")
    assert (cp.hits, cp.misses) == (1, 1)
    now["t"] = 61
    cp.get_history("AAPL", "2026-01-01", "2026-02-01")
    assert cp.misses == 2


def test_cache_returns_copies():
    cp = CachedProvider(SyntheticPriceProvider())
    a = cp.get_history("AAPL", "2026-01-01", "2026-02-01")
    a.iloc[:, 0] = 0
    assert (cp.get_history("AAPL", "2026-01-01", "2026-02-01")["close"] > 0).all()


YAHOO = {"chart": {"result": [{"timestamp": [1761782400, 1761868800, 1761955200],
                               "indicators": {"quote": [{"open": [1, 2, 3], "high": [2, 3, 4], "low": [0.5, 1, 2],
                                                         "close": [1.5, None, 3.5], "volume": [10, 20, 30]}]}}], "error": None}}


def test_parse_yahoo_drops_missing_close():
    df = parse_yahoo_chart(YAHOO)
    assert list(df["close"]) == [1.5, 3.5] and df.index.tz is None


def test_parse_yahoo_errors():
    with pytest.raises(ProviderError):
        parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})
    with pytest.raises(ProviderError):
        parse_yahoo_chart({"chart": {"result": [], "error": None}})


AV = {"Time Series (Daily)": {"2026-10-30": {"1. open": "10", "2. high": "11", "3. low": "9", "4. close": "10.5", "5. volume": "100"},
                              "2026-10-29": {"1. open": "9", "2. high": "10", "3. low": "8", "4. close": "9.5", "5. volume": "90"}}}


def test_parse_alpha_vantage_daily_sorted():
    df = parse_alpha_vantage_daily(AV)
    assert list(df["close"]) == [9.5, 10.5] and df.index.is_monotonic_increasing


def test_parse_alpha_vantage_errors():
    for payload in ({"Note": "rate limit"}, {"Error Message": "bad"}, {}):
        with pytest.raises(ProviderError):
            parse_alpha_vantage_daily(payload)


def test_staleness_is_business_day_aware():
    assert not is_stale("2026-10-30", "2026-10-31")      # Friday close seen on Saturday
    assert not is_stale("2026-10-30", "2026-11-02")      # Monday morning
    assert is_stale("2026-10-20", "2026-10-30")
    assert not is_stale("2026-10-30", "2026-10-29")       # future-dated data is not 'stale'


def test_get_provider():
    assert get_provider("synthetic").is_synthetic
    with pytest.raises(ValueError):
        get_provider("nope")
    with pytest.raises(ValueError):
        get_provider("alphavantage", "")
