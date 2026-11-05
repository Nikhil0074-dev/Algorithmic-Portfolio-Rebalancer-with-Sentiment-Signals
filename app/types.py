"""Small shared types (kept free of heavy imports)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SentimentSignal:
    """Asset-level aggregated sentiment. score is in [-1, 1] or None if unavailable."""
    score: Optional[float]
    n_articles: int = 0
    n_events: int = 0
    confidence: float = 0.0
    n_pos: int = 0
    n_neg: int = 0
    n_neu: int = 0
    n_uncertain: int = 0

    def eligible(self, min_confidence: float, min_events: int) -> bool:
        """A signal may influence decisions only if it has enough confident evidence."""
        return (
            self.score is not None
            and self.n_events >= min_events
            and self.confidence >= min_confidence
        )
