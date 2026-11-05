import math
from pathlib import Path

import pandas as pd
import pytest

from app.services.news import assign_events
from app.services.sentiment import (LexiconFinancialClassifier, aggregate_sentiment, build_sentiment_history,
                                    classify_articles, evaluate_classifier, label_from_probs)

CLF = LexiconFinancialClassifier()
AS_OF = pd.Timestamp("2026-10-30 21:00", tz="UTC")


def mk(rows):
    """rows: (symbol, title, hours_before_as_of, relevance)"""
    recs = []
    for i, (sym, title, hrs, rel) in enumerate(rows):
        t = AS_OF - pd.Timedelta(hours=hrs)
        recs.append({"article_id": f"a{i}", "symbol": sym, "title": title, "publisher": "p", "url": f"https://x.com/{i}",
                     "published_at": t, "available_at": t, "relevance": rel})
    return classify_articles(assign_events(pd.DataFrame(recs)), CLF)


def test_probabilities_normalised_and_labels():
    for text in ["Company beats estimates", "Shares plunge on fraud probe", "Company names new head", ""]:
        p = CLF.classify_one(text)
        assert sum(p.values()) == pytest.approx(1.0)
    assert label_from_probs(CLF.classify_one("Firm raises guidance after record quarter")) == "positive"
    assert label_from_probs(CLF.classify_one("Firm misses estimates and shares slump")) == "negative"
    assert label_from_probs(CLF.classify_one("Firm names new regional head")) == "neutral"


def test_negation_flips_polarity():
    assert label_from_probs(CLF.classify_one("Company did not miss estimates")) == "positive"
    assert label_from_probs(CLF.classify_one("Company fails to deliver growth")) == "negative"


def test_mixed_evidence_is_neutral():
    assert label_from_probs(CLF.classify_one("Revenue beats but guidance misses")) == "neutral"


def test_no_news_means_unavailable():
    sig = aggregate_sentiment(mk([("AAPL", "Apple beats estimates", 1, 1.0)]), "MSFT", AS_OF)
    assert sig.score is None and sig.n_events == 0
    assert aggregate_sentiment(pd.DataFrame(), "AAPL", AS_OF).score is None


def test_score_sign_and_range():
    pos = aggregate_sentiment(mk([("AAPL", "Apple beats estimates as demand surges", 2, 1.0)]), "AAPL", AS_OF)
    neg = aggregate_sentiment(mk([("AAPL", "Apple misses estimates, shares slump", 2, 1.0)]), "AAPL", AS_OF)
    assert 0.5 < pos.score <= 1 and -1 <= neg.score < -0.5


def test_time_decay_prefers_recent_news():
    df = mk([("AAPL", "Apple misses estimates, shares slump", 24 * 10, 1.0),     # old negative
             ("AAPL", "Apple beats estimates as demand surges", 1, 1.0)])         # fresh positive
    assert aggregate_sentiment(df, "AAPL", AS_OF, half_life_hours=24).score > 0.5


def test_half_life_weighting_math():
    df = mk([("AAPL", "Apple beats estimates as demand surges", 0, 1.0),
             ("AAPL", "Apple misses estimates, shares slump", 48, 1.0)])
    s = aggregate_sentiment(df, "AAPL", AS_OF, half_life_hours=48)
    sp, sn = df["score"].iloc[1 if df["title"].iloc[0].startswith("Apple misses") else 0], None
    pos = df[df.title.str.contains("beats")].score.iloc[0]
    neg = df[df.title.str.contains("misses")].score.iloc[0]
    expected = (1.0 * pos + 0.5 * neg) / 1.5
    assert s.score == pytest.approx(expected, rel=1e-6)


def test_duplicates_count_once():
    one = mk([("AAPL", "Apple misses estimates, shares slump", 3, 1.0), ("AAPL", "Apple beats estimates as demand surges", 3, 1.0)])
    dup = mk([("AAPL", "Apple misses estimates, shares slump", 3, 1.0), ("AAPL", "Apple misses estimates, shares slump - Reuters", 2.9, 1.0),
              ("AAPL", "Apple misses estimates, shares slump - Bloomberg", 2.8, 1.0), ("AAPL", "Apple beats estimates as demand surges", 3, 1.0)])
    a, b = aggregate_sentiment(one, "AAPL", AS_OF), aggregate_sentiment(dup, "AAPL", AS_OF)
    assert b.n_events == 2 and b.n_articles == 4
    assert b.score == pytest.approx(a.score, abs=1e-6)       # copies add no extra weight


def test_low_confidence_is_uncertain_not_directional():
    df = mk([("AAPL", "Revenue beats but guidance misses", 1, 1.0)])   # mixed -> low confidence
    sig = aggregate_sentiment(df, "AAPL", AS_OF, min_confidence=0.55)
    assert sig.n_uncertain == 1 and sig.score is None


def test_all_neutral_gives_near_zero():
    df = mk([("AAPL", "Apple names new regional head", 1, 1.0), ("AAPL", "Apple announces date for results", 2, 1.0)])
    assert abs(aggregate_sentiment(df, "AAPL", AS_OF).score) < 0.05


def test_future_articles_are_excluded():
    df = mk([("AAPL", "Apple beats estimates as demand surges", -5, 1.0)])  # published AFTER as_of
    assert aggregate_sentiment(df, "AAPL", AS_OF).score is None


def test_market_news_has_low_relevance_weight():
    df = mk([("AAPL", "Apple beats estimates as demand surges", 1, 1.0), ("MARKET", "Markets slump on rate concerns", 1, 0.3)])
    s = aggregate_sentiment(df, "AAPL", AS_OF).score
    assert s > 0 and s < df[df.symbol == "AAPL"].score.iloc[0]


def test_history_has_no_lookahead():
    day = pd.Timestamp("2026-10-28")
    pub = pd.Timestamp("2026-10-29 10:00", tz="UTC")
    df = classify_articles(assign_events(pd.DataFrame([{
        "article_id": "x", "symbol": "AAPL", "title": "Apple beats estimates as demand surges", "publisher": "p", "url": "u",
        "published_at": pub, "available_at": pub + pd.Timedelta(hours=1), "relevance": 1.0}])), CLF)
    h = build_sentiment_history(df, ["AAPL"], [day, pd.Timestamp("2026-10-29"), pd.Timestamp("2026-10-30")])
    assert h[day]["AAPL"].score is None                              # before publication
    assert h[pd.Timestamp("2026-10-29")]["AAPL"].score is not None   # after publication, same-day close
    assert h[pd.Timestamp("2026-10-30")]["AAPL"].n_events == 1


def test_classifier_evaluation_sanity():
    data = pd.read_csv(Path(__file__).parents[1] / "data" / "labeled_headlines.csv")
    res = evaluate_classifier(CLF, data["text"], data["label"])
    assert res["accuracy"] >= 0.75            # sanity floor on a tiny, author-written set (NOT a benchmark)
    assert sum(sum(r) for r in res["confusion_matrix"]) == len(data)
