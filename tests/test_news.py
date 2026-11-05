import pandas as pd
import pytest

from app.services.market_data import ProviderError
from app.services.news import (SyntheticNewsProvider, assign_events, canonical_url, normalize_title,
                               parse_alpha_vantage_news, relevance_score)


def test_normalize_title_strips_publisher_and_punctuation():
    assert normalize_title("Apple Beats Estimates! - Reuters") == normalize_title("apple beats estimates")


def test_canonical_url_drops_tracking_and_www():
    assert canonical_url("https://www.Example.com/a/b/?utm_source=x&id=3#frag") == "https://example.com/a/b?id=3"


def test_relevance():
    assert relevance_score("Apple unveils new iPhone", "AAPL") == 1.0
    assert relevance_score("Markets slump", "AAPL") == 0.3


def _df(rows):
    return pd.DataFrame([{"article_id": str(i), "symbol": s, "title": t, "publisher": "p", "url": u,
                          "published_at": pd.Timestamp(ts, tz="UTC"), "available_at": pd.Timestamp(ts, tz="UTC"),
                          "relevance": 1.0} for i, (s, t, u, ts) in enumerate(rows)])


def test_duplicates_grouped_and_distinct_kept():
    df = assign_events(_df([
        ("AAPL", "Apple misses estimates, shares slump", "https://a.com/1", "2026-10-29 10:00"),
        ("AAPL", "Apple misses estimates, shares slump - Reuters", "https://b.com/9", "2026-10-29 10:20"),
        ("AAPL", "Apple names new regional head", "https://c.com/3", "2026-10-29 11:00"),
    ]))
    assert df["event_id"].nunique() == 2


def test_same_url_with_tracking_is_same_event():
    df = assign_events(_df([
        ("AAPL", "Totally different words here", "https://a.com/x?utm_source=1", "2026-10-29 10:00"),
        ("AAPL", "Another headline entirely", "https://www.a.com/x/?utm_source=2", "2026-10-29 10:05"),
    ]))
    assert df["event_id"].nunique() == 1


def test_same_headline_far_apart_or_other_symbol_not_merged():
    df = assign_events(_df([
        ("AAPL", "Company beats estimates", "u1", "2026-10-20 10:00"),
        ("AAPL", "Company beats estimates", "u2", "2026-10-29 10:00"),     # 9 days later
        ("MSFT", "Company beats estimates", "u3", "2026-10-29 10:05"),     # other symbol
    ]))
    assert df["event_id"].nunique() == 3


def test_synthetic_news_deterministic_and_range_independent():
    p = SyntheticNewsProvider()
    a = p.get_news(["AAPL", "MSFT"], "2026-09-01", "2026-10-30")
    b = SyntheticNewsProvider().get_news(["AAPL", "MSFT"], "2026-09-01", "2026-10-30")
    pd.testing.assert_frame_equal(a, b)
    sub = p.get_news(["AAPL", "MSFT"], "2026-10-01", "2026-10-30")
    assert set(sub["article_id"]) <= set(a["article_id"])
    assert (a["available_at"] > a["published_at"]).all()
    assert a["published_at"].dt.tz is not None


def test_synthetic_news_contains_duplicates_to_dedupe():
    df = assign_events(SyntheticNewsProvider(duplicate_prob=0.8).get_news(["AAPL"], "2026-08-01", "2026-10-30"))
    assert df["event_id"].nunique() < len(df)


AV = {"feed": [
    {"title": "Nvidia beats estimates", "url": "https://x.com/1", "time_published": "20261030T120000", "source": "Wire",
     "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.9"}, {"ticker": "AAPL", "relevance_score": "0.2"}]},
    {"title": "No timestamp", "url": "https://x.com/2", "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.9"}]},
    {"title": "Bad relevance", "url": "https://x.com/3", "time_published": "20261029T090000", "source": "S",
     "ticker_sentiment": [{"ticker": "MSFT", "relevance_score": "abc"}]},
]}


def test_parse_alpha_vantage_news():
    df = parse_alpha_vantage_news(AV)
    assert len(df) == 3                                   # 2 tickers + MSFT; item without timestamp excluded
    assert set(df["symbol"]) == {"NVDA", "AAPL", "MSFT"}
    assert df.loc[df.symbol == "NVDA", "relevance"].iloc[0] == pytest.approx(0.9)
    assert df.loc[df.symbol == "MSFT", "relevance"].iloc[0] == pytest.approx(0.3)
    assert (df["available_at"] > df["published_at"]).all()


def test_parse_alpha_vantage_rate_limit_and_empty():
    with pytest.raises(ProviderError):
        parse_alpha_vantage_news({"Information": "rate limit reached"})
    assert parse_alpha_vantage_news({"feed": []}).empty
