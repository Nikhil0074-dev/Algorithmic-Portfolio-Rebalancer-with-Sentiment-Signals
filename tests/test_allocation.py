import pytest

from app.engine.allocation import CASH, drift_table, portfolio_value, validate_targets


def test_targets_summing_to_one_accepted(targets):
    validate_targets(targets)


@pytest.mark.parametrize("bad", [
    {"A": 0.5, "CASH": 0.6}, {"A": 0.4, "CASH": 0.4}, {}, {"A": -0.1, "CASH": 1.1}, {"A": float("nan"), "CASH": 1.0},
])
def test_invalid_targets_rejected(bad):
    with pytest.raises(ValueError):
        validate_targets(bad)


def test_tolerance_accepts_tiny_error():
    validate_targets({"A": 0.3333333, "B": 0.3333333, "CASH": 0.3333334})


def test_portfolio_value(holdings, prices):
    assert portfolio_value(holdings, prices, 4000) == pytest.approx(20000)


def test_missing_price_raises(holdings):
    with pytest.raises(ValueError, match="Missing"):
        portfolio_value(holdings, {"AAPL": 1.0}, 0)


def test_drift_table_values(holdings, prices, targets):
    df = drift_table(holdings, prices, 4000, targets).set_index("asset")
    assert df["weight"].sum() == pytest.approx(1.0)
    assert df.loc["AAPL", "weight"] == pytest.approx(0.40)
    assert df.loc["AAPL", "drift_pp"] == pytest.approx(10.0)      # overweight
    assert df.loc["MSFT", "drift_pp"] == pytest.approx(-5.0)      # underweight
    assert df.loc["CASH", "drift_pp"] == pytest.approx(0.0)
    assert df.loc["AAPL", "trade_value"] == pytest.approx(-2000)  # sell
    assert df.loc["MSFT", "trade_value"] == pytest.approx(1000)   # buy
    assert df["trade_value"].sum() == pytest.approx(0, abs=1e-9)


def test_in_range_flag(holdings, prices, targets):
    df = drift_table(holdings, prices, 4000, targets, {"AAPL": (0.25, 0.45), "MSFT": (0.29, 0.31)}).set_index("asset")
    assert bool(df.loc["AAPL", "in_range"]) is True
    assert bool(df.loc["MSFT", "in_range"]) is False


def test_zero_portfolio_value_is_controlled_error(targets):
    with pytest.raises(ValueError, match="positive"):
        drift_table({}, {"AAPL": 1, "MSFT": 1, "NVDA": 1}, 0, targets)


def test_target_asset_not_held_is_included(prices):
    df = drift_table({"AAPL": 10}, prices, 1000, {"AAPL": 0.5, "MSFT": 0.5}).set_index("asset")
    assert df.loc["MSFT", "value"] == 0 and df.loc["MSFT", "trade_value"] > 0
