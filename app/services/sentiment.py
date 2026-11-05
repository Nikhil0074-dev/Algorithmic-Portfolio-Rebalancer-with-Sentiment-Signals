"""Financial sentiment: classifier(s), event grouping, time-weighted aggregation.

Default classifier is a transparent finance lexicon model (no downloads needed).
FinBERT is an optional drop-in (needs `transformers` + `torch` + internet to fetch weights).
Sentiment is an uncertain signal; nothing here predicts prices.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..types import SentimentSignal

LABELS = ("positive", "negative", "neutral")

_POS = {
    "beat": 1.5, "beats": 1.5, "surge": 1.5, "surges": 1.5, "soar": 1.5, "soars": 1.5, "record": 1.0,
    "strong": 1.0, "growth": 1.0, "upgrade": 1.5, "upgraded": 1.5, "upgrades": 1.5, "raises": 1.0,
    "raised": 1.0, "outperform": 1.5, "rally": 1.0, "rallies": 1.0, "profit": 1.0, "profits": 1.0,
    "gains": 1.0, "expands": 1.0, "expansion": 1.0, "approval": 1.0, "approved": 1.0, "wins": 1.0,
    "boost": 1.0, "boosts": 1.0, "optimistic": 1.0, "buyback": 1.0, "exceeds": 1.5, "tops": 1.5,
    "robust": 1.0, "accelerates": 1.0, "rebounds": 1.0, "jumps": 1.5, "climbs": 1.0,
}
_NEG = {
    "miss": 1.5, "misses": 1.5, "missed": 1.5, "plunge": 1.5, "plunges": 1.5, "slump": 1.5,
    "slumps": 1.5, "downgrade": 1.5, "downgraded": 1.5, "downgrades": 1.5, "cuts": 1.0, "cut": 1.0,
    "lawsuit": 1.0, "probe": 1.0, "investigation": 1.0, "recall": 1.0, "layoffs": 1.0, "weak": 1.0,
    "decline": 1.0, "declines": 1.0, "loss": 1.0, "losses": 1.0, "fraud": 1.5, "warns": 1.0,
    "warning": 1.0, "delay": 1.0, "delays": 1.0, "ban": 1.0, "fined": 1.0, "fine": 1.0,
    "antitrust": 1.0, "shortfall": 1.5, "concerns": 1.0, "slowdown": 1.0, "slowing": 1.0,
    "bankruptcy": 1.5, "default": 1.5, "halts": 1.0, "falls": 1.0, "drops": 1.0, "tumbles": 1.5,
    "sinks": 1.5, "lowers": 1.0, "disappoints": 1.5, "selloff": 1.0,
}
_NEGATORS = {"not", "no", "never", "without", "fails", "failed", "denies", "unlikely"}
_TOKEN = re.compile(r"[a-z']+")


class LexiconFinancialClassifier:
    """Rule-based classifier with negation handling. Returns class probabilities."""
    name = "lexicon-financial-v1"

    def __init__(self, scale: float = 2.0, neutral_logit: float = 1.2) -> None:
        self.scale, self.neutral_logit = scale, neutral_logit

    def _scores(self, text: str) -> tuple[float, float]:
        toks = _TOKEN.findall(text.lower())
        pos = neg = 0.0
        for i, t in enumerate(toks):
            w_pos, w_neg = _POS.get(t, 0.0), _NEG.get(t, 0.0)
            if not (w_pos or w_neg):
                continue
            negated = any(x in _NEGATORS for x in toks[max(0, i - 3):i])
            if negated:
                w_pos, w_neg = w_neg, w_pos
            pos += w_pos
            neg += w_neg
        return pos, neg

    def classify_one(self, text: str) -> Dict[str, float]:
        pos, neg = self._scores(text or "")
        logits = np.array([self.scale * pos, self.scale * neg, self.neutral_logit])
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return {"positive": float(p[0]), "negative": float(p[1]), "neutral": float(p[2])}

    def classify(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        return [self.classify_one(t) for t in texts]


class FinBERTClassifier:
    """Optional ProsusAI/finbert wrapper. NOT exercised by the test-suite
    (needs network + large dependencies)."""
    name = "ProsusAI/finbert"

    def __init__(self, batch_size: int = 16) -> None:
        from transformers import pipeline  # lazy import
        self._pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        self.batch_size = batch_size

    def classify(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        out = []
        for res in self._pipe(list(texts), batch_size=self.batch_size, truncation=True):
            out.append({d["label"].lower(): float(d["score"]) for d in res})
        return out


def get_classifier(use_finbert: bool = False):
    if use_finbert:
        try:
            return FinBERTClassifier()
        except Exception as exc:  # pragma: no cover - environment dependent
            import warnings
            warnings.warn(f"FinBERT unavailable ({exc}); falling back to lexicon classifier")
    return LexiconFinancialClassifier()


def label_from_probs(p: Dict[str, float], tie_margin: float = 0.05) -> str:
    if abs(p["positive"] - p["negative"]) < tie_margin and max(p["positive"], p["negative"]) > p["neutral"]:
        return "neutral"  # mixed evidence is not a directional call
    return max(LABELS, key=lambda k: p[k])


def classify_articles(df: pd.DataFrame, classifier) -> pd.DataFrame:
    """Adds p_pos, p_neg, p_neu, label, confidence, score columns."""
    out = df.copy()
    if out.empty:
        for c in ("p_pos", "p_neg", "p_neu", "confidence", "score"):
            out[c] = pd.Series(dtype=float)
        out["label"] = pd.Series(dtype=object)
        return out
    probs = classifier.classify(out["title"].fillna("").tolist())
    out["p_pos"] = [p.get("positive", 0.0) for p in probs]
    out["p_neg"] = [p.get("negative", 0.0) for p in probs]
    out["p_neu"] = [p.get("neutral", 0.0) for p in probs]
    out["label"] = [label_from_probs({"positive": a, "negative": b, "neutral": c})
                    for a, b, c in zip(out["p_pos"], out["p_neg"], out["p_neu"])]
    out["confidence"] = out[["p_pos", "p_neg", "p_neu"]].max(axis=1)
    out["score"] = out["p_pos"] - out["p_neg"]
    return out


def aggregate_sentiment(
    df: pd.DataFrame,
    symbol: str,
    as_of,
    half_life_hours: float = 48.0,
    min_confidence: float = 0.55,
    lookback_hours: float = 24 * 14,
) -> SentimentSignal:
    """S = sum(r*d*s) / sum(r*d) over *events* (duplicates collapsed), using only
    articles whose `available_at` <= as_of. Low-confidence events are counted as
    uncertain and excluded from the weighted mean. Returns score=None if no evidence."""
    if df is None or df.empty:
        return SentimentSignal(score=None)
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    avail = df["available_at"] if "available_at" in df else df["published_at"]
    mask = ((df["symbol"] == symbol) | (df["symbol"] == "MARKET")) & (avail <= as_of) \
        & (avail >= as_of - pd.Timedelta(hours=lookback_hours))
    sub = df[mask]
    if sub.empty:
        return SentimentSignal(score=None)
    sub = sub.assign(_avail=avail[mask])
    # collapse duplicates -> one row per event
    ev = sub.groupby("event_id").agg(
        score=("score", "mean"), confidence=("confidence", "mean"),
        relevance=("relevance", "max"), t=("_avail", "min"), n=("score", "size"),
        label=("label", lambda s: s.mode().iloc[0]),
    )
    lam = math.log(2) / half_life_hours
    age_h = (as_of - ev["t"]).dt.total_seconds() / 3600.0
    decay = np.exp(-lam * age_h.clip(lower=0))
    conf_ok = ev["confidence"] >= min_confidence
    w = ev["relevance"] * decay
    used = ev[conf_ok]
    den = float(w[conf_ok].sum())
    score = float((w[conf_ok] * used["score"]).sum() / den) if den > 1e-12 else None
    return SentimentSignal(
        score=score,
        n_articles=int(ev["n"].sum()),
        n_events=int(len(ev)),
        confidence=float(ev["confidence"].mean()),
        n_pos=int((ev["label"] == "positive").sum()),
        n_neg=int((ev["label"] == "negative").sum()),
        n_neu=int((ev["label"] == "neutral").sum()),
        n_uncertain=int((~conf_ok).sum()),
    )


def build_sentiment_history(
    df: pd.DataFrame, symbols: Iterable[str], dates: Iterable, decision_hour_utc: int = 21, **kw
) -> Dict[pd.Timestamp, Dict[str, SentimentSignal]]:
    """Point-in-time sentiment for each decision date (decision = close of that day)."""
    hist: Dict[pd.Timestamp, Dict[str, SentimentSignal]] = {}
    for d in dates:
        ts = pd.Timestamp(d).normalize().tz_localize("UTC") + pd.Timedelta(hours=decision_hour_utc)
        hist[pd.Timestamp(d).normalize()] = {s: aggregate_sentiment(df, s, ts, **kw) for s in symbols}
    return hist


def evaluate_classifier(classifier, texts: Sequence[str], labels: Sequence[str]) -> Dict[str, object]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    probs = classifier.classify(list(texts))
    pred = [label_from_probs(p) for p in probs]
    p, r, f, _ = precision_recall_fscore_support(labels, pred, labels=list(LABELS), average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(LABELS)).tolist(),
        "labels": list(LABELS), "predictions": pred,
    }
