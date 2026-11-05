"""FastAPI application. Single-user MVP: there is NO authentication (see docs/risk_disclosures.md)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .db import Repository
from .engine.allocation import CASH, drift_table, validate_targets
from .engine.backtester import STRATEGIES, BacktestConfig, backtest
from .engine.rebalancer import RebalanceConfig, rebalance
from .services import pipeline
from .services.market_data import PriceProvider, ProviderError, get_provider
from .services.news import AlphaVantageNewsProvider, NewsProvider, SyntheticNewsProvider
from .services.sentiment import build_sentiment_history, get_classifier

DISCLAIMER = ("Proposals are for manual review only. Sentiment is an uncertain signal, not a price forecast. "
              "Nothing here is investment advice and no orders are placed.")


def _clean(o):
    """Make results JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class PortfolioIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    base_currency: str = "USD"
    cash: float = Field(0.0, ge=0)
    benchmark: Optional[str] = None


class PortfolioPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    base_currency: Optional[str] = None
    cash: Optional[float] = Field(None, ge=0)
    benchmark: Optional[str] = None


class HoldingIn(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    asset_type: str = "stock"
    quantity: float = Field(ge=0)
    average_cost: float = Field(0.0, ge=0)


class ImportIn(BaseModel):
    csv: str


class AllocationIn(BaseModel):
    targets: Dict[str, float]
    ranges: Optional[Dict[str, List[float]]] = None


class RebalanceIn(BaseModel):
    threshold_pp: float = Field(3.0, ge=0)
    min_trade_value: float = Field(50.0, ge=0)
    fee_rate: float = Field(0.001, ge=0)
    fixed_fee: float = Field(0.0, ge=0)
    min_cash_weight: float = Field(0.0, ge=0, lt=1)
    max_weight: Optional[float] = Field(None, gt=0, le=1)
    fractional: bool = True
    max_turnover: Optional[float] = Field(None, gt=0)
    restricted: List[str] = []
    sentiment_mode: str = "off"
    sentiment_k: float = 0.5
    max_tilt: float = Field(0.25, ge=0, lt=1)
    as_of: Optional[str] = None


class BacktestIn(BaseModel):
    targets: Dict[str, float]
    start: str
    end: str
    strategies: List[str] = list(STRATEGIES)
    initial_value: float = Field(100_000.0, gt=0)
    periodic_days: int = Field(21, ge=1)
    threshold_pp: float = Field(5.0, ge=0)
    fee_rate: float = Field(0.001, ge=0)
    sentiment_k: float = 0.5
    max_tilt: float = Field(0.25, ge=0, lt=1)
    benchmark: Optional[str] = None


def create_app(db_path: str = ":memory:", provider: Optional[PriceProvider] = None,
               news_provider: Optional[NewsProvider] = None, classifier=None,
               settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Algorithmic Portfolio Rebalancer", version=__version__, description=DISCLAIMER)
    repo = Repository(db_path)
    provider = provider or get_provider(settings.provider, settings.alphavantage_api_key)
    if news_provider is None:
        news_provider = (AlphaVantageNewsProvider(settings.alphavantage_api_key)
                         if settings.news_provider == "alphavantage" else SyntheticNewsProvider())
    classifier = classifier or get_classifier(settings.use_finbert)
    app.state.repo = repo

    def need(pid: int):
        p = repo.get_portfolio(pid)
        if not p:
            raise HTTPException(404, "Portfolio not found")
        return p

    def state(pid: int):
        p = need(pid)
        qty = {h["symbol"]: h["quantity"] for h in repo.list_holdings(pid)}
        return p, qty

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "price_provider": provider.name,
                "price_data_is_synthetic": provider.is_synthetic, "news_data_is_synthetic": news_provider.is_synthetic,
                "classifier": classifier.name}

    @app.post("/api/portfolios", status_code=201)
    def create_portfolio(body: PortfolioIn):
        pid = repo.create_portfolio(body.name, body.base_currency, body.cash, body.benchmark)
        return repo.get_portfolio(pid)

    @app.get("/api/portfolios")
    def list_portfolios():
        return repo.list_portfolios()

    @app.get("/api/portfolios/{pid}")
    def get_portfolio(pid: int):
        p = need(pid)
        version, targets = repo.get_targets(pid)
        return {**p, "holdings": repo.list_holdings(pid), "target_version": version,
                "targets": {t["symbol"]: t["weight"] for t in targets}}

    @app.patch("/api/portfolios/{pid}")
    def patch_portfolio(pid: int, body: PortfolioPatch):
        need(pid)
        repo.update_portfolio(pid, **body.model_dump())
        return repo.get_portfolio(pid)

    @app.delete("/api/portfolios/{pid}", status_code=204)
    def delete_portfolio(pid: int):
        need(pid)
        repo.delete_portfolio(pid)

    @app.post("/api/portfolios/{pid}/holdings", status_code=201)
    def add_holding(pid: int, body: HoldingIn):
        need(pid)
        sym = body.symbol.strip().upper()
        if not sym.replace("-", "").replace(".", "").isalnum():
            raise HTTPException(422, "Invalid symbol")
        if sym == CASH:
            repo.update_portfolio(pid, cash=body.quantity)
        else:
            repo.upsert_holding(pid, sym, body.asset_type, body.quantity, body.average_cost)
        return {"symbol": sym, "quantity": body.quantity}

    @app.get("/api/portfolios/{pid}/holdings")
    def get_holdings(pid: int):
        p = need(pid)
        return {"cash": p["cash"], "holdings": repo.list_holdings(pid)}

    @app.delete("/api/portfolios/{pid}/holdings/{symbol}", status_code=204)
    def remove_holding(pid: int, symbol: str):
        need(pid)
        if not repo.delete_holding(pid, symbol.upper()):
            raise HTTPException(404, "Holding not found")

    @app.post("/api/portfolios/{pid}/import")
    def import_csv(pid: int, body: ImportIn):
        need(pid)
        try:
            rows = pipeline.parse_holdings_csv(body.csv)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        repo.replace_portfolio(pid, rows)
        return {"imported": len(rows)}

    @app.put("/api/portfolios/{pid}/allocations")
    def put_allocations(pid: int, body: AllocationIn):
        need(pid)
        targets = {k.upper(): v for k, v in body.targets.items()}
        try:
            validate_targets(targets)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        ranges = {k.upper(): tuple(v) for k, v in (body.ranges or {}).items() if len(v) == 2}
        return {"version": repo.set_targets(pid, targets, ranges), "targets": targets}

    def _market(pid, qty, as_of, extra_symbols=()):
        symbols = set(qty) | set(extra_symbols)
        prices, dates, stale, errors = pipeline.latest_prices(provider, symbols, as_of, settings.max_price_age_bdays)
        return prices, dates, stale, errors

    @app.get("/api/portfolios/{pid}/drift")
    def drift(pid: int, as_of: Optional[str] = None):
        p, qty = state(pid)
        version, trows = repo.get_targets(pid)
        if not trows:
            raise HTTPException(422, "Set target allocations first")
        targets = {t["symbol"]: t["weight"] for t in trows}
        ranges = {t["symbol"]: (t["min_weight"], t["max_weight"]) for t in trows if t["min_weight"] is not None}
        when = pd.Timestamp(as_of) if as_of else pipeline.last_business_day()
        prices, dates, stale, errors = _market(pid, qty, when, targets)
        if errors:
            raise HTTPException(502, {"message": "Price data unavailable", "errors": errors})
        try:
            df = drift_table(qty, prices, p["cash"], targets, ranges)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return _clean({"as_of": str(when.date()), "price_dates": dates, "stale": sorted(stale),
                       "synthetic_data": provider.is_synthetic, "drift": df.to_dict("records")})

    @app.get("/api/assets/{symbol}/prices")
    def asset_prices(symbol: str, start: Optional[str] = None, end: Optional[str] = None):
        end_ts = pd.Timestamp(end) if end else pipeline.last_business_day()
        start_ts = pd.Timestamp(start) if start else end_ts - pd.Timedelta(days=90)
        try:
            df = provider.get_history(symbol.upper(), start_ts, end_ts)
        except ProviderError as exc:
            raise HTTPException(502, str(exc))
        return _clean({"symbol": symbol.upper(), "provider": provider.name, "synthetic": provider.is_synthetic,
                       "prices": [{"date": str(i.date()), "close": r["close"]} for i, r in df.iterrows()]})

    @app.get("/api/assets/{symbol}/sentiment")
    def asset_sentiment(symbol: str, days: int = 30):
        end = pipeline.last_business_day()
        start = end - pd.Timedelta(days=min(max(days, 1), 365))
        sym = symbol.upper()
        df = pipeline.load_news(news_provider, classifier, [sym], start, end)
        sig = pipeline.aggregate_sentiment(df, sym, end.tz_localize("UTC") + pd.Timedelta(hours=23))
        arts = df[df["symbol"] == sym].tail(20)
        return _clean({"symbol": sym, "synthetic_news": news_provider.is_synthetic, "model": classifier.name,
                       "signal": sig.__dict__, "latest": [
                           {"title": r.title, "published_at": r.published_at.isoformat(), "label": r.label,
                            "confidence": r.confidence, "url": r.url} for r in arts.itertuples()]})

    @app.post("/api/portfolios/{pid}/rebalance")
    def rebalance_endpoint(pid: int, body: RebalanceIn):
        p, qty = state(pid)
        version, trows = repo.get_targets(pid)
        if not trows:
            raise HTTPException(422, "Set target allocations first")
        targets = {t["symbol"]: t["weight"] for t in trows}
        when = pd.Timestamp(body.as_of) if body.as_of else pipeline.last_business_day()
        prices, dates, stale, errors = _market(pid, qty, when, targets)
        if errors:
            raise HTTPException(502, {"message": "Price data unavailable", "errors": errors})
        try:
            cfg = RebalanceConfig(**{k: v for k, v in body.model_dump().items() if k != "as_of"})
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc))
        signals = {}
        if cfg.sentiment_mode != "off":
            syms = [s for s in set(qty) | set(targets) if s != CASH]
            news = pipeline.load_news(news_provider, classifier, syms, when - pd.Timedelta(days=21), when)
            signals = pipeline.current_signals(news, syms, when.tz_localize("UTC") + pd.Timedelta(hours=21))
        try:
            res = rebalance({k: v for k, v in qty.items()}, prices, p["cash"], targets, cfg, signals, stale)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        trades = _clean(res.trades.to_dict("records"))
        summary = {"portfolio_value": res.portfolio_value, "cash_before": res.cash_before, "cash_after": res.cash_after,
                   "turnover": res.turnover, "total_fees": res.total_fees, "warnings": res.warnings,
                   "incomplete": res.incomplete, "targets_used": res.targets_used}
        market = {"as_of": str(when.date()), "prices": prices, "price_dates": dates, "stale": sorted(stale),
                  "synthetic": provider.is_synthetic}
        rid = repo.save_run(pid, version, body.model_dump(), market, _clean({k: v.__dict__ for k, v in signals.items()}),
                            _clean(summary), trades, "incomplete" if res.incomplete else "proposed")
        return {"run_id": rid, "status": "incomplete" if res.incomplete else "proposed", "summary": _clean(summary),
                "trades": trades, "market": market, "disclaimer": DISCLAIMER}

    @app.get("/api/rebalance-runs/{rid}")
    def get_run(rid: int):
        run = repo.get_run(rid)
        if not run:
            raise HTTPException(404, "Run not found")
        return run

    @app.post("/api/backtests", status_code=201)
    def run_backtest(body: BacktestIn):
        try:
            validate_targets({k.upper(): v for k, v in body.targets.items()})
            bad = [s for s in body.strategies if s not in STRATEGIES]
            if bad:
                raise ValueError(f"Unknown strategies: {bad}")
            targets = {k.upper(): v for k, v in body.targets.items()}
            start, end = pd.Timestamp(body.start), pd.Timestamp(body.end)
            if end <= start:
                raise ValueError("end must be after start")
            syms = [s for s in targets if s != CASH]
            prices = pipeline.price_matrix(provider, syms, start, end)
            bench = None
            if body.benchmark:
                bench = pipeline.price_matrix(provider, [body.benchmark.upper()], start, end).iloc[:, 0].reindex(prices.index).ffill()
            warnings = []
            hist = None
            if "sentiment" in body.strategies:
                news = pipeline.load_news(news_provider, classifier, syms, start - pd.Timedelta(days=21), end)
                hist = build_sentiment_history(news, syms, prices.index)
                if news_provider.is_synthetic:
                    warnings.append("News is synthetic and unrelated to prices: the sentiment result tests plumbing only and is not evidence.")
            if provider.is_synthetic:
                warnings.append("Prices are synthetic; results are not historical market evidence.")
            out = {}
            for s in body.strategies:
                cfg = BacktestConfig(strategy=s, targets=targets, initial_value=body.initial_value,
                                     periodic_days=body.periodic_days, threshold_pp=body.threshold_pp,
                                     rebalance=RebalanceConfig(min_trade_value=0.0, fee_rate=body.fee_rate,
                                                               sentiment_mode="adjust", sentiment_k=body.sentiment_k,
                                                               max_tilt=body.max_tilt))
                r = backtest(prices, cfg, hist, bench)
                out[s] = {"metrics": r.metrics, "equity": {str(d.date()): v for d, v in r.equity.items()}}
        except (ValueError, ProviderError) as exc:
            raise HTTPException(422, str(exc))
        result = _clean({"strategies": out, "warnings": warnings})
        bid = repo.save_backtest(body.model_dump(), result)
        return {"id": bid, **result}

    @app.get("/api/backtests/{bid}")
    def get_backtest(bid: int):
        b = repo.get_backtest(bid)
        if not b:
            raise HTTPException(404, "Backtest not found")
        return b

    return app


def get_app() -> FastAPI:  # `uvicorn app.api:get_app --factory`
    return create_app(db_path=Settings.from_env().db_path)
