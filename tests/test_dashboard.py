"""Smoke-test every dashboard page with Streamlit's AppTest (no browser needed)."""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "dashboard.py")
os.environ["APP_AS_OF"] = "2026-10-30"
PAGES = ["Overview", "Portfolio", "Sentiment Explorer", "Rebalancing Lab", "Backtesting Lab", "Settings"]


def fresh():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = fresh()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.header[0].value == page


def test_synthetic_data_banner_is_shown():
    at = fresh()
    assert any("SYNTHETIC" in i.value for i in at.info)


def test_invalid_targets_block_pages_with_clear_error():
    at = fresh()
    at.sidebar.radio[0].set_value("Rebalancing Lab").run()
    at.session_state["targets"] = at.session_state["targets"].assign(target_pct=[10.0, 10.0, 10.0, 10.0, 10.0])
    at.run()
    assert not at.exception
    assert any("Target allocation problem" in e.value for e in at.error)


def test_backtest_runs_end_to_end():
    at = fresh()
    at.sidebar.radio[0].set_value("Backtesting Lab").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert len(at.dataframe) >= 1
