import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.db import Repository
from app.services.news import SyntheticNewsProvider
from app.services.sentiment import LexiconFinancialClassifier
from tests.conftest import FixedProvider

PRICES = {"AAPL": 200.0, "MSFT": 400.0, "NVDA": 100.0, "BND": 70.0}
CSV = "symbol,asset_type,quantity,average_cost\nAAPL,stock,40,180\nMSFT,stock,12.5,400\nNVDA,stock,30,110\nCASH,cash,4000,1\n"
TARGETS = {"AAPL": 0.3, "MSFT": 0.3, "NVDA": 0.2, "CASH": 0.2}
ASOF = "2026-10-30"


def make_client(last_date="2026-10-30"):
    app = create_app(provider=FixedProvider(PRICES, last_date), news_provider=SyntheticNewsProvider(),
                     classifier=LexiconFinancialClassifier(), settings=Settings())
    return TestClient(app)


@pytest.fixture
def client():
    return make_client()


@pytest.fixture
def pid(client):
    pid = client.post("/api/portfolios", json={"name": "Growth"}).json()["id"]
    assert client.post(f"/api/portfolios/{pid}/import", json={"csv": CSV}).status_code == 200
    assert client.put(f"/api/portfolios/{pid}/allocations", json={"targets": TARGETS}).status_code == 200
    return pid


# ---------------------------------------------------------------- repository
def test_repo_versioned_targets_and_cascade():
    r = Repository()
    pid = r.create_portfolio("p")
    assert r.get_targets(pid) == (0, [])
    assert r.set_targets(pid, {"A": 0.5, "CASH": 0.5}) == 1
    assert r.set_targets(pid, {"A": 0.6, "CASH": 0.4}) == 2
    v, rows = r.get_targets(pid)
    assert v == 2 and {x["symbol"]: x["weight"] for x in rows} == {"A": 0.6, "CASH": 0.4}
    r.upsert_holding(pid, "A", "stock", 5, 10)
    r.upsert_holding(pid, "A", "stock", 7, 10)             # upsert, not duplicate
    assert [h["quantity"] for h in r.list_holdings(pid)] == [7]
    assert len(r.list_transactions(pid)) == 2
    assert r.delete_portfolio(pid)
    assert r.list_holdings(pid) == [] and r.get_targets(pid) == (0, [])


def test_repo_update_whitelists_fields():
    r = Repository()
    pid = r.create_portfolio("p")
    r.update_portfolio(pid, name="q", evil="x'; DROP TABLE portfolios;--")
    assert r.get_portfolio(pid)["name"] == "q"


# ---------------------------------------------------------------- portfolios & holdings
def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["classifier"] == LexiconFinancialClassifier.name


def test_portfolio_crud(client):
    assert client.post("/api/portfolios", json={"name": ""}).status_code == 422
    assert client.post("/api/portfolios", json={"name": "x", "cash": -5}).status_code == 422
    pid = client.post("/api/portfolios", json={"name": "A", "cash": 100}).json()["id"]
    assert client.patch(f"/api/portfolios/{pid}", json={"name": "B"}).json()["name"] == "B"
    assert len(client.get("/api/portfolios").json()) == 1
    assert client.delete(f"/api/portfolios/{pid}").status_code == 204
    for call in (client.get, client.delete):
        assert call(f"/api/portfolios/{pid}").status_code == 404
    assert client.patch("/api/portfolios/999", json={"name": "x"}).status_code == 404


def test_holdings_validation_and_cash(client):
    pid = client.post("/api/portfolios", json={"name": "A"}).json()["id"]
    assert client.post(f"/api/portfolios/{pid}/holdings", json={"symbol": "AAPL", "quantity": 3}).status_code == 201
    assert client.post(f"/api/portfolios/{pid}/holdings", json={"symbol": "AAPL", "quantity": -1}).status_code == 422
    assert client.post(f"/api/portfolios/{pid}/holdings", json={"symbol": "A;B", "quantity": 1}).status_code == 422
    assert client.post(f"/api/portfolios/{pid}/holdings", json={"symbol": "cash", "quantity": 250}).status_code == 201
    data = client.get(f"/api/portfolios/{pid}/holdings").json()
    assert data["cash"] == 250 and [h["symbol"] for h in data["holdings"]] == ["AAPL"]
    assert client.delete(f"/api/portfolios/{pid}/holdings/AAPL").status_code == 204
    assert client.delete(f"/api/portfolios/{pid}/holdings/AAPL").status_code == 404
    assert client.post("/api/portfolios/999/holdings", json={"symbol": "A", "quantity": 1}).status_code == 404


@pytest.mark.parametrize("csv", [
    pytest.param("ticker,qty\nAAPL,1", id="missing-columns"),
    pytest.param("symbol,quantity\nAAPL,-3", id="negative-quantity"),
    pytest.param("symbol,quantity\nAAPL,1\nAAPL,2", id="duplicate-symbol"),
    pytest.param("symbol,quantity\nAAPL,abc", id="invalid-quantity"),
    pytest.param("symbol,quantity\n;;;,1", id="invalid-symbol"),
    pytest.param("symbol,quantity\n" + "AAPL,1\n" * 5 + "x" * 300_000, id="oversized-csv"),
])
def test_csv_import_rejects_bad_input(client, csv):
    pid = client.post("/api/portfolios", json={"name": "A"}).json()["id"]
    assert client.post(f"/api/portfolios/{pid}/import", json={"csv": csv}).status_code == 422


def test_csv_import_rejects_non_finite_values(client):
    pid = client.post("/api/portfolios", json={"name": "A"}).json()["id"]
    for csv in (
        "symbol,quantity\nAAPL,inf",
        "symbol,quantity,average_cost\nAAPL,1,inf",
        "symbol,quantity\nAAPL,-inf",
    ):
        assert client.post(f"/api/portfolios/{pid}/import", json={"csv": csv}).status_code == 422


def test_csv_import_replaces_existing_holdings(client):
    pid = client.post("/api/portfolios", json={"name": "A"}).json()["id"]
    client.post(f"/api/portfolios/{pid}/import", json={"csv": CSV})
    replacement = "symbol,quantity,average_cost\nMSFT,2,400\nCASH,25,1\n"
    response = client.post(f"/api/portfolios/{pid}/import", json={"csv": replacement})
    assert response.json() == {"imported": 2}
    portfolio = client.get(f"/api/portfolios/{pid}").json()
    assert portfolio["cash"] == 25
    assert [(h["symbol"], h["quantity"]) for h in portfolio["holdings"]] == [("MSFT", 2)]


def test_csv_import_good(client):
    pid = client.post("/api/portfolios", json={"name": "A"}).json()["id"]
    assert client.post(f"/api/portfolios/{pid}/import", json={"csv": CSV}).json() == {"imported": 4}
    p = client.get(f"/api/portfolios/{pid}").json()
    assert p["cash"] == 4000 and len(p["holdings"]) == 3


# ---------------------------------------------------------------- allocations / drift
def test_allocations_validation_and_versioning(client):
    pid = client.post("/api/portfolios", json={"name": "A"}).json()["id"]
    assert client.put(f"/api/portfolios/{pid}/allocations", json={"targets": {"AAPL": 0.9}}).status_code == 422
    assert client.put(f"/api/portfolios/{pid}/allocations", json={"targets": {"AAPL": 0.5, "CASH": 0.5}}).json()["version"] == 1
    assert client.put(f"/api/portfolios/{pid}/allocations", json={"targets": {"aapl": 0.4, "CASH": 0.6}}).json()["version"] == 2
    assert client.get(f"/api/portfolios/{pid}").json()["targets"] == {"AAPL": 0.4, "CASH": 0.6}


def test_drift_requires_targets(client):
    pid = client.post("/api/portfolios", json={"name": "A"}).json()["id"]
    assert client.get(f"/api/portfolios/{pid}/drift").status_code == 422


def test_drift_values(client, pid):
    body = client.get(f"/api/portfolios/{pid}/drift", params={"as_of": ASOF}).json()
    rows = {r["asset"]: r for r in body["drift"]}
    assert sum(r["weight"] for r in rows.values()) == pytest.approx(1.0)
    assert body["stale"] == [] and rows["CASH"]["value"] == 4000


def test_drift_flags_stale_prices():
    c = make_client(last_date="2026-10-20")
    pid = c.post("/api/portfolios", json={"name": "A"}).json()["id"]
    c.post(f"/api/portfolios/{pid}/import", json={"csv": CSV})
    c.put(f"/api/portfolios/{pid}/allocations", json={"targets": TARGETS})
    assert set(c.get(f"/api/portfolios/{pid}/drift", params={"as_of": ASOF}).json()["stale"]) == {"AAPL", "MSFT", "NVDA"}


def test_unknown_symbol_gives_502_not_crash(client):
    pid = client.post("/api/portfolios", json={"name": "A"}).json()["id"]
    client.post(f"/api/portfolios/{pid}/holdings", json={"symbol": "ZZZZ", "quantity": 1})
    client.put(f"/api/portfolios/{pid}/allocations", json={"targets": {"ZZZZ": 1.0}})
    r = client.get(f"/api/portfolios/{pid}/drift", params={"as_of": ASOF})
    assert r.status_code == 502 and "ZZZZ" in str(r.json())


# ---------------------------------------------------------------- rebalance
def test_rebalance_baseline_and_persistence(client, pid):
    body = client.post(f"/api/portfolios/{pid}/rebalance", json={"as_of": ASOF, "fee_rate": 0, "min_trade_value": 0}).json()
    assert body["status"] == "proposed" and "investment advice" in body["disclaimer"].lower()
    t = {x["symbol"]: x for x in body["trades"]}
    assert t["AAPL"]["action"] == "SELL" and t["MSFT"]["action"] == "BUY"
    assert body["summary"]["cash_after"] >= 0
    run = client.get(f"/api/rebalance-runs/{body['run_id']}").json()
    assert len(run["trades"]) == 3 and run["target_version"] == 1 and run["config"]["fee_rate"] == 0
    assert run["market"]["prices"]["AAPL"] > 0 and run["summary"]["portfolio_value"] > 0


def test_rebalance_with_sentiment_modes(client, pid):
    base = client.post(f"/api/portfolios/{pid}/rebalance", json={"as_of": ASOF}).json()
    rev = client.post(f"/api/portfolios/{pid}/rebalance", json={"as_of": ASOF, "sentiment_mode": "review"}).json()
    assert [t["quantity"] for t in base["trades"]] == [t["quantity"] for t in rev["trades"]]
    adj = client.post(f"/api/portfolios/{pid}/rebalance", json={"as_of": ASOF, "sentiment_mode": "adjust", "sentiment_k": 1.0}).json()
    assert adj["status"] == "proposed" and sum(adj["summary"]["targets_used"].values()) == pytest.approx(1.0)


def test_rebalance_input_errors(client, pid):
    assert client.post(f"/api/portfolios/{pid}/rebalance", json={"sentiment_mode": "bogus", "as_of": ASOF}).status_code == 422
    assert client.post(f"/api/portfolios/{pid}/rebalance", json={"threshold_pp": -1}).status_code == 422
    new = client.post("/api/portfolios", json={"name": "empty"}).json()["id"]
    assert client.post(f"/api/portfolios/{new}/rebalance", json={}).status_code == 422
    assert client.post("/api/portfolios/999/rebalance", json={}).status_code == 404
    assert client.get("/api/rebalance-runs/999").status_code == 404


def test_rebalance_with_stale_data_is_incomplete():
    c = make_client(last_date="2026-10-20")
    pid = c.post("/api/portfolios", json={"name": "A"}).json()["id"]
    c.post(f"/api/portfolios/{pid}/import", json={"csv": CSV})
    c.put(f"/api/portfolios/{pid}/allocations", json={"targets": TARGETS})
    body = c.post(f"/api/portfolios/{pid}/rebalance", json={"as_of": ASOF}).json()
    assert body["status"] == "incomplete" and all(t["action"] == "HOLD" for t in body["trades"])


# ---------------------------------------------------------------- prices / sentiment / backtests
def test_prices_and_sentiment_endpoints(client):
    p = client.get("/api/assets/aapl/prices", params={"start": "2026-10-01", "end": ASOF}).json()
    assert p["symbol"] == "AAPL" and len(p["prices"]) > 10 and p["synthetic"] is False
    s = client.get("/api/assets/AAPL/sentiment").json()
    assert s["synthetic_news"] is True and "signal" in s
    assert client.get("/api/assets/NOPE/prices").status_code == 502


BT = {"targets": {"AAPL": 0.4, "MSFT": 0.4, "CASH": 0.2}, "start": "2026-02-02", "end": "2026-10-30"}


def test_backtest_flow(client):
    r = client.post("/api/backtests", json=BT)
    assert r.status_code == 201
    body = r.json()
    assert set(body["strategies"]) == {"buy_hold", "periodic", "threshold", "sentiment"}
    assert any("synthetic" in w.lower() for w in body["warnings"])       # synthetic news must be flagged
    for v in body["strategies"].values():
        assert len(v["equity"]) > 100 and "max_drawdown" in v["metrics"]
    got = client.get(f"/api/backtests/{body['id']}").json()
    assert got["config"]["start"] == "2026-02-02" and got["result"]["strategies"].keys() == body["strategies"].keys()


@pytest.mark.parametrize("patch", [
    {"strategies": ["magic"]}, {"end": "2026-01-01"}, {"targets": {"AAPL": 0.5}}, {"fee_rate": -1}, {"initial_value": 0},
    {"targets": {"AAPL": 0.5, "ZZZ": 0.5}},
])
def test_backtest_rejects_bad_input(client, patch):
    assert client.post("/api/backtests", json={**BT, **patch}).status_code == 422


def test_backtest_404(client):
    assert client.get("/api/backtests/999").status_code == 404
