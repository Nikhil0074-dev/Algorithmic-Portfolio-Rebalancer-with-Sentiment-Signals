"""Environment-based settings. API keys are read from the environment only."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    provider: str = "synthetic"          # synthetic | yahoo | alphavantage
    news_provider: str = "synthetic"     # synthetic | alphavantage
    alphavantage_api_key: str = ""
    db_path: str = "rebalancer.db"
    use_finbert: bool = False
    max_price_age_bdays: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(
            provider=e("PRICE_PROVIDER", "synthetic").lower(),
            news_provider=e("NEWS_PROVIDER", "synthetic").lower(),
            alphavantage_api_key=e("ALPHAVANTAGE_API_KEY", ""),
            db_path=e("DB_PATH", "rebalancer.db"),
            use_finbert=e("USE_FINBERT", "0") == "1",
            max_price_age_bdays=int(e("MAX_PRICE_AGE_BDAYS", "3")),
        )
