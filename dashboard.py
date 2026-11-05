"""Streamlit dashboard: decision support only. No orders are ever placed."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app.services.market_data import ProviderError

from app.config import Settings
from app.engine import metrics as M
from app.engine.allocation import CASH, drift_table, validate_targets
from app.engine.backtester import STRATEGIES, BacktestConfig, backtest
from app.engine.rebalancer import RebalanceConfig, rebalance
from app.engine.risk import concentration_flags, volatility_flag
from app.services import pipeline
from app.services.market_data import get_provider
from app.services.news import AlphaVantageNewsProvider, SyntheticNewsProvider
from app.services.sentiment import (aggregate_sentiment, build_sentiment_history, get_classifier)

ROOT = Path(__file__).parent
PALETTE = ["#b8e05a", "#f2a65a", "#55b7a6", "#aa8fce", "#e47764", "#65766f"]
DISCLAIMER = ("Decision support only - not investment advice. Sentiment is an uncertain signal, not a price forecast. "
              "Backtests do not predict future results. No orders are placed.")
DEFAULT_TARGETS = {"AAPL": 25.0, "MSFT": 25.0, "NVDA": 20.0, "BND": 20.0, "CASH": 10.0}

st.set_page_config(page_title="Fieldnotes | Portfolio Studio", page_icon="◒", layout="wide")


# ----------------------------------------------------------------------------- visual system
def apply_visual_system() -> None:
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root {
        --paper: #f3f2ec; --surface: #fbfaf6; --ink: #1d2822; --muted: #778078;
        --line: #dedfd5; --moss: #344d3d; --lime: #c3e66a; --orange: #e9a16b;
      }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
      .stApp { background: var(--paper); }
      [data-testid="stHeader"] { background: rgba(243,242,236,.92); }
      [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.045em !important; color: var(--ink) !important; }
      h1 { font-size: 2.15rem !important; font-weight: 700 !important; }
      h2 { font-size: 1.42rem !important; font-weight: 700 !important; }
      [data-testid="stSidebar"] { background: var(--moss); border-right: 0; }
      [data-testid="stSidebar"] * { color: #f6f5ee; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 9px; padding: .36rem .58rem; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(195,230,106,.12); }
      [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] { background: rgba(195,230,106,.18); }
      [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] p { color: var(--lime) !important; font-weight: 700; }
      [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 13px; padding: 17px 19px; box-shadow: 0 3px 12px #29342b08; }
      [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .78rem !important; text-transform: uppercase; letter-spacing: .09em; }
      [data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 1.65rem; }
      [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--surface); }
      [data-testid="stPlotlyChart"] { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 7px; }
      [data-testid="stExpander"], [data-testid="stFileUploader"] { border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
      .stButton > button, .stDownloadButton > button { border-radius: 9px; border: 1px solid var(--moss); background: var(--moss); color: white; font-weight: 600; transition: transform .15s ease, background .15s ease; }
      .stButton > button:hover, .stDownloadButton > button:hover { background: #49624f; color: white; border-color: #49624f; transform: translateY(-1px); }
      [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input { background: var(--surface); border-radius: 8px; }
      hr { border-color: var(--line); }
      .fn-brand { padding: 12px 7px 24px; border-bottom: 1px solid #ffffff27; margin-bottom: 23px; }
      .fn-brand-mark { display:inline-grid; place-items:center; width:36px; height:36px; border-radius:11px; background:var(--lime); color:var(--moss)!important; font-size:23px; margin-right:10px; vertical-align:middle; }
      .fn-brand-name { font:700 18px 'Manrope',sans-serif; letter-spacing:-.04em; vertical-align:middle; }
      .fn-brand-sub { color:#c1cbbd!important; font:10px 'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; margin:11px 0 0 2px; }
      .fn-intro { display:flex; justify-content:space-between; align-items:flex-end; gap:20px; padding:18px 22px; margin: -3px 0 24px; background:var(--moss); border-radius:15px; color:#f7f5ea; }
      .fn-intro-kicker { color:var(--lime); font:11px 'DM Mono',monospace; letter-spacing:.15em; text-transform:uppercase; margin-bottom:8px; }
      .fn-intro-title { color:#fffdf4; font:700 clamp(20px,3vw,28px) 'Manrope',sans-serif; letter-spacing:-.045em; }
      .fn-intro-copy { color:#d1d8cd; font-size:13px; margin-top:5px; }
      .fn-intro-icon { color:var(--lime); font-size:31px; line-height:1; }
      .fn-ribbon { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); margin-bottom:23px; padding:0 2px 13px; color:var(--muted); font:10px 'DM Mono',monospace; letter-spacing:.09em; text-transform:uppercase; }
      .fn-live { color:#527542; }
      .fn-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#a9cf58; margin-right:7px; box-shadow:0 0 0 3px #a9cf5827; }
      [data-testid="stCaptionContainer"] { color:var(--muted); }
      @media (max-width: 700px) { [data-testid="stMainBlockContainer"] {padding:1rem 1rem 3rem;} .fn-intro {padding:15px;}.fn-intro-icon{display:none;} }
    </style>
    """, unsafe_allow_html=True)


def page_intro(kicker: str, title: str, description: str, icon: str) -> None:
    st.markdown(
        f'<div class="fn-intro"><div><div class="fn-intro-kicker">{kicker}</div>'
        f'<div class="fn-intro-title">{title}</div><div class="fn-intro-copy">{description}</div></div>'
        f'<div class="fn-intro-icon">{icon}</div></div>',
        unsafe_allow_html=True,
    )



# ----------------------------------------------------------------------------- resources
@st.cache_resource
def get_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def get_price_provider():
    s = get_settings()
    return get_provider(s.provider, s.alphavantage_api_key)


@st.cache_resource
def get_news_provider():
    s = get_settings()
    return AlphaVantageNewsProvider(s.alphavantage_api_key) if s.news_provider == "alphavantage" else SyntheticNewsProvider()


@st.cache_resource
def get_nlp():
    return get_classifier(get_settings().use_finbert)


@st.cache_data(show_spinner=False)
def cached_news(symbols: tuple, start: str, end: str) -> pd.DataFrame:
    return pipeline.load_news(get_news_provider(), get_nlp(), list(symbols), start, end)


@st.cache_data(show_spinner=False)
def cached_history(symbols: tuple, start: str, end: str) -> pd.DataFrame:
    return pipeline.price_matrix(get_price_provider(), list(symbols), start, end)


# ----------------------------------------------------------------------------- state
def init_state() -> None:
    if "holdings" in st.session_state:
        return
    rows = pipeline.parse_holdings_csv((ROOT / "data" / "sample_holdings.csv").read_text())
    st.session_state.holdings = pd.DataFrame(
        [{"symbol": r["symbol"], "quantity": r["quantity"], "average_cost": r["average_cost"]} for r in rows if r["symbol"] != CASH])
    st.session_state.cash = next((r["quantity"] for r in rows if r["symbol"] == CASH), 0.0)
    st.session_state.targets = pd.DataFrame({"asset": list(DEFAULT_TARGETS), "target_pct": list(DEFAULT_TARGETS.values())})


def get_qty() -> dict:
    df = st.session_state.holdings.dropna(subset=["symbol"])
    return {str(r.symbol).strip().upper(): float(r.quantity) for r in df.itertuples() if str(r.symbol).strip() and pd.notna(r.quantity)}


def get_targets() -> dict:
    df = st.session_state.targets.dropna(subset=["asset"])
    return {str(r.asset).strip().upper(): float(r.target_pct) / 100.0 for r in df.itertuples() if str(r.asset).strip()}


def require_valid_setup(as_of):
    qty, targets = get_qty(), get_targets()
    try:
        validate_targets(targets)
    except ValueError as exc:
        st.error(f"Target allocation problem: {exc}. Fix it on the Portfolio page.")
        st.stop()
    symbols = set(qty) | {a for a in targets if a != CASH}
    prices, dates, stale, errors = pipeline.latest_prices(get_price_provider(), symbols, as_of, get_settings().max_price_age_bdays)
    if errors:
        st.error("Price data unavailable for: " + ", ".join(f"{k} ({v})" for k, v in errors.items()))
        st.stop()
    if stale:
        st.warning("Stale price data for " + ", ".join(sorted(stale)) + ". These assets are excluded from proposals.")
    return qty, targets, prices, dates, stale


def signals_for(symbols, as_of):
    start, end = (as_of - pd.Timedelta(days=21)).strftime("%Y-%m-%d"), as_of.strftime("%Y-%m-%d")
    news = cached_news(tuple(sorted(symbols)), start, end)
    ts = as_of.tz_localize("UTC") + pd.Timedelta(hours=21)
    return news, pipeline.current_signals(news, symbols, ts)


def banner():
    notes = []
    if get_price_provider().is_synthetic:
        notes.append("prices are SYNTHETIC")
    if get_news_provider().is_synthetic:
        notes.append("news is SYNTHETIC (unrelated to prices)")
    if notes:
        st.info("Demo data: " + " and ".join(notes) + ". Results illustrate the software, not real markets.")


def fig_style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=12, r=12, t=48, b=12),
                      legend=dict(orientation="h", y=-0.2), colorway=PALETTE,
                      paper_bgcolor="#fbfaf6", plot_bgcolor="#fbfaf6",
                      font=dict(family="DM Sans, sans-serif", color="#526058", size=11),
                      title_font=dict(family="Manrope, sans-serif", size=15, color="#25362a"),
                      xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(gridcolor="#e8e8df", zerolinecolor="#d8ddd0"))
    return fig


def pct(x):
    return f"{x:.1%}"


# ----------------------------------------------------------------------------- pages
def page_overview(as_of):
    st.header("Overview")
    page_intro("FIELDNOTE  /  01", "A quieter view of your capital", "A daily snapshot of allocation, drift and portfolio signals.", "◒")
    qty, targets, prices, dates, stale = require_valid_setup(as_of)
    cash = float(st.session_state.cash)
    dt = drift_table(qty, prices, cash, targets)
    total = dt["value"].sum()
    cost = sum(float(r.average_cost) * float(r.quantity) for r in st.session_state.holdings.itertuples()
               if str(r.symbol).upper() in qty and pd.notna(r.average_cost))
    invested = total - cash
    worst = dt.loc[dt["drift_pp"].abs().idxmax()]
    c = st.columns(4)
    c[0].metric("Portfolio value", f"${total:,.0f}")
    c[1].metric("Unrealised P&L vs cost", f"${invested - cost:,.0f}", f"{(invested / cost - 1):.1%}" if cost > 0 else None)
    c[2].metric("Largest drift", f"{worst.asset} {worst.drift_pp:+.1f}pp")
    c[3].metric("Latest price date", max(dates.values()) if dates else "n/a")

    l, r = st.columns([3, 2])
    fig = go.Figure([go.Bar(name="Current", x=dt["asset"], y=dt["weight"] * 100, marker_color=PALETTE[0]),
                     go.Bar(name="Target", x=dt["asset"], y=dt["target_weight"] * 100, marker_color=PALETTE[1])])
    fig.update_layout(barmode="group", title="Current vs target allocation (%)", yaxis_title="% of portfolio")
    l.plotly_chart(fig_style(fig), use_container_width=True)
    pie = go.Figure(go.Pie(labels=dt["asset"], values=dt["value"], hole=0.55, marker=dict(colors=PALETTE)))
    pie.update_layout(title="Current composition")
    r.plotly_chart(fig_style(pie), use_container_width=True)

    show = dt.assign(weight=dt["weight"].map(pct), target_weight=dt["target_weight"].map(pct)).rename(columns={
        "asset": "Asset", "value": "Value ($)", "weight": "Current", "target_weight": "Target", "drift_pp": "Drift (pp)",
        "target_value": "Target value ($)", "trade_value": "Gap to target ($)", "in_range": "In range"})
    st.dataframe(show.style.format({"Value ($)": "{:,.0f}", "Drift (pp)": "{:+.2f}", "Target value ($)": "{:,.0f}",
                                    "Gap to target ($)": "{:+,.0f}"}), use_container_width=True, hide_index=True)

    st.subheader("Risk indicators")
    flags = concentration_flags(dict(zip(dt["asset"], dt["weight"])), 0.35)
    for f in flags:
        st.warning("Concentration: " + f)
    vol_rows = []
    for s in qty:
        h = cached_history((s,), (as_of - pd.Timedelta(days=200)).strftime("%Y-%m-%d"), as_of.strftime("%Y-%m-%d"))[s]
        v = volatility_flag(h)
        vol_rows.append({"Asset": s, "Recent vol (21d)": v["recent"], "Long-run vol": v["long_run"],
                         "Flag": "elevated" if v["flag"] else "normal"})
    st.dataframe(pd.DataFrame(vol_rows).style.format({"Recent vol (21d)": "{:.1%}", "Long-run vol": "{:.1%}"}, na_rep="n/a"),
                 use_container_width=True, hide_index=True)
    news, sig = signals_for([s for s in qty], as_of)
    alerts = [s for s, v in sig.items() if v.score is not None and abs(v.score) >= 0.4 and v.eligible(0.55, 2)]
    st.caption(f"Sentiment review alerts: {', '.join(alerts) if alerts else 'none'}")


def page_portfolio(as_of):
    st.header("Portfolio")
    page_intro("WORKSPACE  /  02", "Shape the portfolio you mean to hold", "Edit positions, import a ledger and tune target weights in one place.", "⌘")
    up = st.file_uploader("Import holdings CSV (symbol, quantity, optional asset_type, average_cost)", type="csv")
    if up is not None and st.button("Apply import"):
        try:
            rows = pipeline.parse_holdings_csv(up.getvalue().decode("utf-8", errors="replace"))
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state.holdings = pd.DataFrame([{"symbol": r["symbol"], "quantity": r["quantity"], "average_cost": r["average_cost"]}
                                                      for r in rows if r["symbol"] != CASH])
            st.session_state.cash = next((r["quantity"] for r in rows if r["symbol"] == CASH), 0.0)
            st.success(f"Imported {len(rows)} rows.")
    l, r = st.columns(2)
    with l:
        st.subheader("Holdings")
        st.session_state.holdings = st.data_editor(st.session_state.holdings, num_rows="dynamic", use_container_width=True,
                                                   key="ed_holdings", column_config={
                                                       "quantity": st.column_config.NumberColumn(min_value=0.0),
                                                       "average_cost": st.column_config.NumberColumn(min_value=0.0)})
        st.session_state.cash = st.number_input("Cash ($)", min_value=0.0, value=float(st.session_state.cash), step=100.0)
    with r:
        st.subheader("Target allocation (%)")
        st.session_state.targets = st.data_editor(st.session_state.targets, num_rows="dynamic", use_container_width=True,
                                                  key="ed_targets", column_config={"target_pct": st.column_config.NumberColumn(min_value=0.0, max_value=100.0)})
        total = float(st.session_state.targets["target_pct"].fillna(0).sum())
        (st.success if abs(total - 100) < 1e-6 else st.error)(f"Targets sum to {total:.2f}% (must be 100%)")
    qty = get_qty()
    if qty:
        st.subheader("Position values")
        prices, dates, stale, errors = pipeline.latest_prices(get_price_provider(), qty, as_of)
        rows = [{"Symbol": s, "Quantity": q, "Price": prices.get(s), "Value ($)": q * prices[s] if s in prices else None,
                 "Price date": dates.get(s, "missing")} for s, q in qty.items()]
        st.dataframe(pd.DataFrame(rows).style.format({"Quantity": "{:,.2f}", "Price": "{:,.2f}", "Value ($)": "{:,.0f}"}, na_rep="n/a"),
                     use_container_width=True, hide_index=True)


def page_sentiment(as_of):
    st.header("Sentiment Explorer")
    page_intro("SIGNALS  /  03", "Read the tone, not the tea leaves", "Explore event-based headlines as a cautious, time-aware context signal.", "✳")
    banner()
    qty, targets, prices, dates, stale = require_valid_setup(as_of)
    symbols = sorted(set(qty) | {a for a in targets if a != CASH})
    news, sig = signals_for(symbols, as_of)
    rows = [{"Asset": s, "Score": v.score, "Events": v.n_events, "Articles": v.n_articles, "Positive": v.n_pos,
             "Negative": v.n_neg, "Neutral": v.n_neu, "Uncertain": v.n_uncertain, "Avg confidence": v.confidence} for s, v in sig.items()]
    df = pd.DataFrame(rows)
    l, r = st.columns([2, 3])
    with l:
        bar = go.Figure(go.Bar(x=df["Asset"], y=df["Score"], marker_color=[PALETTE[2] if (x or 0) >= 0 else PALETTE[4] for x in df["Score"]]))
        bar.update_layout(title="Aggregated sentiment score (-1 to +1)", yaxis=dict(range=[-1, 1]))
        st.plotly_chart(fig_style(bar, 320), use_container_width=True)
    with r:
        st.dataframe(df.style.format({"Score": "{:+.2f}", "Avg confidence": "{:.2f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)
    st.caption("Duplicate stories are grouped into events. Scores use a 48h half-life and ignore low-confidence events. Model: " + get_nlp().name)

    asset = st.selectbox("Asset", symbols)
    days = pd.bdate_range(as_of - pd.Timedelta(days=30), as_of)
    hist = build_sentiment_history(news, [asset], days)
    trend = pd.Series({d: h[asset].score for d, h in hist.items()})
    line = go.Figure(go.Scatter(x=trend.index, y=trend.values, mode="lines+markers", line=dict(color=PALETTE[0]), connectgaps=False))
    line.update_layout(title=f"{asset} sentiment trend (point-in-time)", yaxis=dict(range=[-1, 1]))
    st.plotly_chart(fig_style(line, 300), use_container_width=True)
    arts = news[news["symbol"].isin([asset, "MARKET"])].sort_values("published_at", ascending=False).head(15)
    st.dataframe(arts[["published_at", "symbol", "title", "publisher", "label", "confidence", "url"]].rename(columns={"published_at": "Published (UTC)"}),
                 use_container_width=True, hide_index=True)


def rebalance_inputs():
    c = st.columns(4)
    thr = c[0].slider("Drift threshold (pp)", 0.0, 15.0, 3.0, 0.5)
    mt = c[1].number_input("Min trade ($)", 0.0, 10000.0, 50.0, 10.0)
    fee = c[2].number_input("Fee rate (%)", 0.0, 5.0, 0.10, 0.05) / 100
    mc = c[3].slider("Min cash (%)", 0.0, 50.0, 5.0, 1.0) / 100
    c = st.columns(4)
    mw = c[0].slider("Max position weight (%)", 10.0, 100.0, 40.0, 5.0) / 100
    mturn = c[1].slider("Max turnover (%)", 5.0, 200.0, 100.0, 5.0) / 100
    frac = c[2].checkbox("Fractional shares", True)
    mode = c[3].selectbox("Sentiment mode", ["review", "adjust", "off"], help="review = flag only; adjust = bounded tilt of targets")
    k, tilt = 0.5, 0.25
    if mode == "adjust":
        c = st.columns(2)
        k = c[0].slider("Sentiment strength k", 0.0, 2.0, 0.5, 0.1)
        tilt = c[1].slider("Max tilt (relative)", 0.0, 0.5, 0.25, 0.05)
    return dict(threshold_pp=thr, min_trade_value=mt, fee_rate=fee, min_cash_weight=mc, max_weight=mw, max_turnover=mturn,
                fractional=frac, sentiment_mode=mode, sentiment_k=k, max_tilt=tilt)


def page_rebalance(as_of):
    st.header("Rebalancing Lab")
    page_intro("DECISION ROOM  /  04", "From drift to a reviewable plan", "Compare a disciplined baseline with bounded controls before considering any proposal.", "↗")
    banner()
    qty, targets, prices, dates, stale = require_valid_setup(as_of)
    kw = rebalance_inputs()
    cash = float(st.session_state.cash)
    symbols = sorted(set(qty) | {a for a in targets if a != CASH})
    sig = {}
    if kw["sentiment_mode"] != "off":
        _, sig = signals_for(symbols, as_of)
    try:
        base = rebalance(qty, prices, cash, targets, RebalanceConfig(**{**kw, "sentiment_mode": "off"}), stale=stale)
        prop = rebalance(qty, prices, cash, targets, RebalanceConfig(**kw), sentiment=sig, stale=stale)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    for w in prop.warnings:
        st.warning(w)
    if prop.incomplete:
        st.error("Proposal is INCOMPLETE because some price data is stale or missing.")
    c = st.columns(4)
    c[0].metric("Portfolio value", f"${prop.portfolio_value:,.0f}")
    c[1].metric("Turnover", pct(prop.turnover))
    c[2].metric("Estimated fees", f"${prop.total_fees:,.2f}")
    c[3].metric("Cash after", f"${prop.cash_after:,.0f}")

    t = prop.trades.copy()
    t["base_qty"] = base.trades["quantity"].values
    t["vs baseline"] = (t["quantity"] - t["base_qty"]).round(4)
    view = t[["symbol", "action", "quantity", "price", "value", "fee", "current_weight", "target_weight", "post_weight",
              "sentiment_score", "sentiment_relation", "review_priority", "vs baseline"]].rename(columns={
        "current_weight": "current", "target_weight": "target", "post_weight": "after", "sentiment_score": "sentiment",
        "sentiment_relation": "signal", "review_priority": "review"})
    st.dataframe(view.style.format({"quantity": "{:,.3f}", "price": "{:,.2f}", "value": "{:+,.0f}", "fee": "{:.2f}", "current": "{:.1%}",
                                    "target": "{:.1%}", "after": "{:.1%}", "sentiment": "{:+.2f}", "vs baseline": "{:+.3f}"}, na_rep="-"),
                 use_container_width=True, hide_index=True)
    fig = go.Figure([go.Bar(name="Current", x=t["symbol"], y=t["current_weight"] * 100), go.Bar(name="Target used", x=t["symbol"], y=t["target_weight"] * 100),
                     go.Bar(name="After trades", x=t["symbol"], y=t["post_weight"] * 100)])
    fig.update_layout(barmode="group", title="Allocation before / target / after (%)")
    st.plotly_chart(fig_style(fig, 320), use_container_width=True)
    with st.expander("Why each trade? (explanations)", expanded=True):
        for r in t.itertuples():
            tag = {"high": "🔴", "medium": "🟠"}.get(r.review_priority, "⚪")
            st.markdown(f"{tag} **{r.symbol} - {r.action}**: {r.rationale}")
    st.download_button("Export proposal (CSV)", t.drop(columns=["base_qty"]).to_csv(index=False), "proposed_trades.csv", "text/csv")
    st.caption(DISCLAIMER)


def page_backtest(as_of):
    st.header("Backtesting Lab")
    page_intro("HISTORY LAB  /  05", "Stress-test the rules against history", "Compare strategy paths with costs, drawdowns and execution timing in view.", "⌁")
    banner()
    qty, targets, *_ = require_valid_setup(as_of)
    c = st.columns(4)
    start = c[0].date_input("Start", (as_of - pd.Timedelta(days=500)).date())
    end = c[1].date_input("End", as_of.date())
    fee = c[2].number_input("Fee rate (%)", 0.0, 5.0, 0.10, 0.05) / 100
    thr = c[3].slider("Threshold (pp)", 1.0, 15.0, 5.0, 0.5)
    c = st.columns(4)
    per = c[0].number_input("Periodic every N days", 1, 252, 21)
    k = c[1].slider("Sentiment k", 0.0, 2.0, 0.5, 0.1)
    tilt = c[2].slider("Max tilt", 0.0, 0.5, 0.25, 0.05)
    bench = c[3].text_input("Benchmark", "SPY").strip().upper()
    strategies = st.multiselect("Strategies", list(STRATEGIES), default=list(STRATEGIES))
    if not st.button("Run backtest", type="primary"):
        st.caption("Choose settings and press Run. " + DISCLAIMER)
        return
    if pd.Timestamp(end) <= pd.Timestamp(start) + pd.Timedelta(days=30):
        st.error("Choose at least ~30 days between start and end.")
        return
    syms = tuple(sorted(a for a in targets if a != CASH))
    try:
        prices = cached_history(syms, str(start), str(end))
        bser = cached_history((bench,), str(start), str(end))[bench].reindex(prices.index).ffill() if bench else None
        hist = None
        if "sentiment" in strategies:
            news = cached_news(syms, (pd.Timestamp(start) - pd.Timedelta(days=21)).strftime("%Y-%m-%d"), str(end))
            hist = build_sentiment_history(news, syms, prices.index)
        results = {}
        for s in strategies:
            cfg = BacktestConfig(strategy=s, targets=targets, periodic_days=int(per), threshold_pp=thr, rebalance=RebalanceConfig(
                min_trade_value=0.0, fee_rate=fee, sentiment_mode="adjust", sentiment_k=k, max_tilt=tilt))
            results[s] = backtest(prices, cfg, hist, bser)
    except (ValueError, ProviderError) as exc:
        st.error(f"Backtest failed: {exc}")
        return
    if get_news_provider().is_synthetic and "sentiment" in results:
        st.warning("News is synthetic and independent of prices, so the sentiment strategy result is a plumbing test, not evidence.")
    eq = go.Figure()
    dd = go.Figure()
    for i, (s, r) in enumerate(results.items()):
        eq.add_trace(go.Scatter(x=r.equity.index, y=r.equity, name=s, line=dict(color=PALETTE[i % 6])))
        dd.add_trace(go.Scatter(x=r.equity.index, y=M.drawdown_series(r.equity) * 100, name=s, line=dict(color=PALETTE[i % 6])))
    eq.update_layout(title="Portfolio value ($)")
    dd.update_layout(title="Drawdown (%)")
    a, b = st.columns(2)
    a.plotly_chart(fig_style(eq, 340), use_container_width=True)
    b.plotly_chart(fig_style(dd, 340), use_container_width=True)
    table = pd.DataFrame({s: r.metrics for s, r in results.items()}).T
    table.index.name = "strategy"
    st.dataframe(table.rename(columns={"cumulative_return": "cum. return", "annualized_return": "ann. return", "annualized_volatility": "ann. vol",
                                       "max_drawdown": "max drawdown", "sharpe_ratio": "Sharpe", "transaction_costs": "costs ($)",
                                       "n_rebalances": "rebalances", "tracking_error": "tracking err."}).style.format({
        "cum. return": "{:.1%}", "ann. return": "{:.1%}", "ann. vol": "{:.1%}", "max drawdown": "{:.1%}", "Sharpe": "{:.2f}",
        "turnover": "{:.2f}", "costs ($)": "{:,.0f}", "rebalances": "{:.0f}", "tracking err.": "{:.1%}"}, na_rep="-"), use_container_width=True)
    first = next(iter(results.values()))
    w = results.get("threshold", first).weights
    area = go.Figure([go.Scatter(x=w.index, y=w[c] * 100, name=c, stackgroup="one") for c in w.columns])
    area.update_layout(title="Allocation history (%) - threshold strategy" if "threshold" in results else "Allocation history (%)")
    st.plotly_chart(fig_style(area, 300), use_container_width=True)
    st.caption(f"Sharpe uses a 0% risk-free rate; 252-day year; {fee:.2%} fee on every trade; 1-day execution lag. " + DISCLAIMER)


def page_settings(as_of):
    st.header("Settings")
    page_intro("PREFERENCES  /  06", "Know what powers the workspace", "Review data sources, model selection and the boundaries of this tool.", "⚙")
    s = get_settings()
    st.table(pd.DataFrame({"Setting": ["Price provider", "News provider", "Sentiment model", "Max price age (business days)", "Database"],
                           "Value": [get_price_provider().name, get_news_provider().name, get_nlp().name, s.max_price_age_bdays, s.db_path]}))
    st.markdown("Configure through environment variables (`PRICE_PROVIDER`, `NEWS_PROVIDER`, `ALPHAVANTAGE_API_KEY`, `USE_FINBERT`). "
                "API keys are read server-side only and are never shown in the UI.")
    st.subheader("Responsible use")
    st.markdown(f"- {DISCLAIMER}\n- Every proposal requires manual review; brokerage execution is intentionally not implemented.\n"
                "- The default demo data is synthetic; see `docs/risk_disclosures.md`.")


PAGES = {"Overview": page_overview, "Portfolio": page_portfolio, "Sentiment Explorer": page_sentiment,
         "Rebalancing Lab": page_rebalance, "Backtesting Lab": page_backtest, "Settings": page_settings}


def main():
    apply_visual_system()
    init_state()
    with st.sidebar:
        st.markdown('<div class="fn-brand"><span class="fn-brand-mark">◒</span><span class="fn-brand-name">Fieldnotes</span><div class="fn-brand-sub">Portfolio studio · 2026</div></div>', unsafe_allow_html=True)
        st.title("Workspace")
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
        default_day = pd.Timestamp(os.environ["APP_AS_OF"]) if os.environ.get("APP_AS_OF") else pipeline.last_business_day()
        as_of = pd.Timestamp(st.date_input("Data as of", default_day.date()))
        st.caption(f"Price source: {get_price_provider().name}")
        st.caption(DISCLAIMER)
    st.markdown('<div class="fn-ribbon"><span><span class="fn-dot"></span>Fieldnotes / decision-support workspace</span><span class="fn-live">No orders · human review always</span></div>', unsafe_allow_html=True)
    if page == "Overview":
        banner()
    PAGES[page](as_of)


main()
