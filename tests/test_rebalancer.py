import math

import pytest

from app.engine.allocation import CASH
from app.engine.rebalancer import RebalanceConfig, apply_sentiment_tilt, rebalance
from tests.conftest import sig

NOFEE = dict(fee_rate=0.0, min_trade_value=0.0)


def run(holdings, prices, targets, cash=4000, **kw):
    sentiment = kw.pop("sentiment", None)
    stale = kw.pop("stale", None)
    return rebalance(holdings, prices, cash, targets, RebalanceConfig(**kw), sentiment, stale)


def trade(res, sym):
    return res.trades.set_index("symbol").loc[sym]


def test_baseline_reaches_targets_without_fees(holdings, prices, targets):
    r = run(holdings, prices, targets, **NOFEE)
    assert trade(r, "AAPL").action == "SELL" and trade(r, "MSFT").action == "BUY"
    for s in ("AAPL", "MSFT", "NVDA"):
        assert trade(r, s).post_weight == pytest.approx(targets[s], abs=1e-9)
    assert r.cash_after == pytest.approx(0.2 * 20000)
    assert r.turnover == pytest.approx((2000 + 1000 + 1000) / 20000)


def test_value_is_conserved_with_fees(holdings, prices, targets):
    r = run(holdings, prices, targets, fee_rate=0.001, min_trade_value=0)
    post_pos = sum((holdings[s] + trade(r, s).quantity) * prices[s] for s in holdings)
    assert post_pos + r.cash_after == pytest.approx(20000 - r.total_fees)
    assert r.total_fees == pytest.approx(0.001 * 4000)


def test_threshold_blocks_small_drift(holdings, prices, targets):
    r = run(holdings, prices, targets, threshold_pp=6.0, **NOFEE)
    assert trade(r, "AAPL").action == "SELL"           # 10pp drift
    assert trade(r, "MSFT").action == "HOLD"           # 5pp drift < 6
    assert "threshold" in trade(r, "MSFT").rationale


def test_min_trade_size(holdings, prices, targets):
    r = run(holdings, prices, targets, threshold_pp=0, fee_rate=0, min_trade_value=1500)
    assert trade(r, "AAPL").action == "SELL" and trade(r, "MSFT").action == "HOLD"


def test_never_sell_more_than_held(prices):
    r = run({"AAPL": 1.0}, prices, {"AAPL": 0.0, "MSFT": 0.0, "CASH": 1.0}, cash=0, **NOFEE)
    assert trade(r, "AAPL").quantity >= -1.0 - 1e-12


def test_insufficient_cash_scales_buys(prices):
    # all-in AAPL, targets want MSFT but selling is restricted -> only cash buys
    r = run({"AAPL": 50.0}, prices, {"AAPL": 0.0, "MSFT": 1.0}, cash=500, restricted={"AAPL"}, **NOFEE)
    assert trade(r, "MSFT").value <= 500 + 1e-6
    assert r.cash_after >= -1e-9
    assert any("cash" in w.lower() for w in r.warnings)


def test_cash_never_negative_with_fees(holdings, prices, targets):
    r = run(holdings, prices, targets, fee_rate=0.01, fixed_fee=2.0, min_trade_value=0)
    assert r.cash_after >= -1e-6


def test_min_cash_reserve_is_respected(holdings, prices):
    t = {"AAPL": 0.3, "MSFT": 0.3, "NVDA": 0.4}   # target cash 0
    r = run(holdings, prices, t, min_cash_weight=0.10, **NOFEE)
    assert r.cash_after >= 0.10 * r.portfolio_value - 1e-6


def test_max_weight_caps_target_and_forces_sell(prices):
    # AAPL is 60% of the portfolio, within 3pp of an (unrealistic) 60% target, but max_weight=40%
    h = {"AAPL": 30.0, "MSFT": 10.0}              # 6000 / 4000, cash 0 -> AAPL 60%
    r = run(h, prices, {"AAPL": 0.6, "MSFT": 0.4}, cash=0, max_weight=0.4, threshold_pp=50, **NOFEE)
    t = trade(r, "AAPL")
    assert t.action == "SELL" and t.post_weight <= 0.4 + 1e-9
    assert "max weight" in t.rationale


def test_turnover_limit(holdings, prices, targets):
    r = run(holdings, prices, targets, max_turnover=0.10, **NOFEE)
    assert r.turnover <= 0.10 + 1e-9
    assert any("turnover" in w.lower() for w in r.warnings)


def test_restricted_asset_never_traded(holdings, prices, targets):
    r = run(holdings, prices, targets, restricted={"AAPL"}, **NOFEE)
    assert trade(r, "AAPL").action == "HOLD" and "restricted" in trade(r, "AAPL").rationale


def test_stale_price_excluded_and_flagged(holdings, prices, targets):
    r = run(holdings, prices, targets, stale={"MSFT"}, **NOFEE)
    assert trade(r, "MSFT").action == "HOLD"
    assert r.incomplete and any("stale" in w for w in r.warnings)


def test_whole_shares_only(holdings, prices, targets):
    r = run(holdings, prices, targets, fractional=False, **NOFEE)
    for q in r.trades["quantity"]:
        assert q == math.floor(abs(q)) * (1 if q >= 0 else -1)
    assert r.cash_after >= -1e-9


def test_missing_price_raises(holdings, targets):
    with pytest.raises(ValueError, match="price"):
        rebalance(holdings, {"AAPL": 1.0}, 0, targets)


def test_zero_value_portfolio_raises(targets, prices):
    with pytest.raises(ValueError, match="positive"):
        rebalance({}, prices, 0, targets)


def test_invalid_targets_raise(holdings, prices):
    with pytest.raises(ValueError):
        rebalance(holdings, prices, 4000, {"AAPL": 0.9})


def test_rebalancing_is_idempotent(holdings, prices, targets):
    r1 = run(holdings, prices, targets, **NOFEE)
    new = {s: holdings[s] + trade(r1, s).quantity for s in holdings}
    r2 = run(new, prices, targets, cash=r1.cash_after, **NOFEE)
    assert (r2.trades["action"] == "HOLD").all()


def test_deterministic(holdings, prices, targets):
    a = run(holdings, prices, targets).trades.to_dict("records")
    b = run(holdings, prices, targets).trades.to_dict("records")
    assert a == b


# ------------------------------------------------------------ sentiment behaviour
def test_sentiment_off_ignores_signals(holdings, prices, targets):
    base = run(holdings, prices, targets)
    with_sig = run(holdings, prices, targets, sentiment={"NVDA": sig(0.9), "AAPL": sig(-0.9)})
    assert base.trades["quantity"].tolist() == with_sig.trades["quantity"].tolist()


def test_review_mode_never_changes_quantities(holdings, prices, targets):
    base = run(holdings, prices, targets)
    rev = run(holdings, prices, targets, sentiment_mode="review", sentiment={"AAPL": sig(0.9), "MSFT": sig(-0.9)})
    assert base.trades["quantity"].tolist() == rev.trades["quantity"].tolist()


def test_review_flags_conflict_and_aligned(holdings, prices, targets):
    r = run(holdings, prices, targets, sentiment_mode="review",
            sentiment={"AAPL": sig(0.8), "MSFT": sig(0.8)})   # AAPL: SELL into good news; MSFT: BUY with good news
    assert trade(r, "AAPL").sentiment_relation == "conflict" and trade(r, "AAPL").review_priority == "high"
    assert trade(r, "MSFT").sentiment_relation == "aligned" and trade(r, "MSFT").review_priority == "medium"


def test_weak_or_thin_sentiment_is_uncertain(holdings, prices, targets):
    r = run(holdings, prices, targets, sentiment_mode="review",
            sentiment={"AAPL": sig(0.9, n=1), "MSFT": sig(0.9, conf=0.3)})
    assert trade(r, "AAPL").sentiment_relation == "uncertain" and trade(r, "MSFT").sentiment_relation == "uncertain"


def test_tilt_is_bounded_and_preserves_noncash_total(targets):
    cfg = RebalanceConfig(sentiment_mode="adjust", sentiment_k=5.0, max_tilt=0.25)
    adj, mult = apply_sentiment_tilt(targets, {"AAPL": sig(1.0), "MSFT": sig(-1.0)}, cfg)
    assert adj["CASH"] == targets["CASH"]
    assert sum(adj.values()) == pytest.approx(1.0)
    assert adj["AAPL"] > targets["AAPL"] > adj["MSFT"]
    assert adj["AAPL"] / targets["AAPL"] < 1.25 + 0.1   # clip then renormalise stays near the cap
    assert adj["NVDA"] != 0


def test_tilt_with_ineligible_signals_is_noop(targets):
    cfg = RebalanceConfig(sentiment_mode="adjust")
    adj, _ = apply_sentiment_tilt(targets, {"AAPL": sig(1.0, n=1), "MSFT": sig(None)}, cfg)
    assert adj == pytest.approx(targets)


def test_adjust_moves_destination_toward_positive_sentiment(holdings, prices, targets):
    base = run(holdings, prices, targets, **NOFEE)
    adj = run(holdings, prices, targets, sentiment_mode="adjust", sentiment_k=1.0,
              sentiment={"MSFT": sig(0.9), "NVDA": sig(-0.9)}, **NOFEE)
    assert trade(adj, "MSFT").target_weight > trade(base, "MSFT").target_weight
    assert trade(adj, "NVDA").target_weight < trade(base, "NVDA").target_weight
    assert trade(adj, "MSFT").base_target_weight == pytest.approx(0.3)
    assert sum(adj.targets_used.values()) == pytest.approx(1.0)


def test_sentiment_alone_cannot_trigger_a_trade(prices):
    h = {"AAPL": 40.0, "MSFT": 20.0}                       # exactly on target (8000/8000 of 16000 incl cash 0)
    t = {"AAPL": 0.5, "MSFT": 0.5}
    r = run(h, prices, t, cash=0, sentiment_mode="adjust", sentiment_k=5.0, threshold_pp=3.0,
            sentiment={"AAPL": sig(1.0), "MSFT": sig(-1.0)}, **NOFEE)
    assert (r.trades["action"] == "HOLD").all()


@pytest.mark.parametrize("kw", [
    {"sentiment_mode": "bogus"}, {"threshold_pp": -1}, {"min_cash_weight": 1.0}, {"max_weight": 0},
    {"max_turnover": 0}, {"max_tilt": 1.0}, {"fee_rate": -0.1},
])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        RebalanceConfig(**kw)


def test_cash_reserve_limits_buys_when_sells_are_blocked(prices):
    # AAPL (restricted) can't be sold, so buys can only use cash ABOVE the 5% reserve
    r = run({"AAPL": 50.0}, prices, {"AAPL": 0.0, "MSFT": 1.0}, cash=1000, restricted={"AAPL"}, min_cash_weight=0.05, **NOFEE)
    assert r.cash_after >= 0.05 * r.portfolio_value - 1e-6
    assert trade(r, "MSFT").value == pytest.approx(1000 - 0.05 * 11000)
