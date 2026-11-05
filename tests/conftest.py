import numpy as np
import pandas as pd
import pytest

from app.services.market_data import PriceProvider
from app.types import SentimentSignal


class FixedProvider(PriceProvider):
    """Deterministic prices for API/integration tests: constant per symbol with a gentle ramp."""
    name = "fixed"
    is_synthetic = False

    def __init__(self, prices, last_date="2026-10-30"):
        self.prices, self.last_date = prices, pd.Timestamp(last_date)

    def get_history(self, symbol, start, end):
        if symbol not in self.prices:
            from app.services.market_data import ProviderError
            raise ProviderError(f"unknown symbol {symbol}")
        idx = pd.bdate_range(pd.Timestamp(start), min(pd.Timestamp(end), self.last_date))
        base = self.prices[symbol]
        days = np.array([(d - pd.Timestamp('2026-01-01')).days for d in idx])
        df = pd.DataFrame({"close": base * (1 + 0.0002 * days)}, index=idx)
        df.index.name = "date"
        return df


@pytest.fixture
def prices():
    return {"AAPL": 200.0, "MSFT": 400.0, "NVDA": 100.0}


@pytest.fixture
def targets():
    return {"AAPL": 0.3, "MSFT": 0.3, "NVDA": 0.2, "CASH": 0.2}


@pytest.fixture
def holdings():
    # values: AAPL 8000, MSFT 5000, NVDA 3000, cash 4000 -> V = 20000
    return {"AAPL": 40.0, "MSFT": 12.5, "NVDA": 30.0}


def sig(score, n=5, conf=0.8):
    return SentimentSignal(score=score, n_articles=n, n_events=n, confidence=conf)
