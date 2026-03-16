"""app/main.py — Portfolio Risk Analyzer Dashboard"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data_loader import DataLoader
from src.risk_engine import RiskEngine
from src.optimizer import PortfolioOptimizer
from src.simulation import MonteCarloSimulator

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
}
.stApp {
    background: #08090e;
    color: #d1d5db;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0f17 !important;
    border-right: 1px solid #1c2033 !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 12px !important;
    color: #6b7280 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
section[data-testid="stSidebar"] h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #60a5fa !important;
}
section[data-testid="stSidebar"] label {
    font-size: 11px !important;
    color: #6b7280 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.05em;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1.5rem;
    background: #0d0f17;
    border-bottom: 1px solid #1c2033;
    margin-bottom: 1.5rem;
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #f9fafb;
    letter-spacing: -0.02em;
}
.topbar-logo span { color: #3b82f6; }
.topbar-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    color: #60a5fa;
    letter-spacing: 0.08em;
}
.topbar-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #374151;
    letter-spacing: 0.05em;
}

/* ── Ticker tape ── */
.ticker-tape {
    background: #0d0f17;
    border-top: 1px solid #1c2033;
    border-bottom: 1px solid #1c2033;
    padding: 6px 0;
    overflow: hidden;
    white-space: nowrap;
    margin-bottom: 1.5rem;
}
.ticker-inner {
    display: inline-block;
    animation: ticker 30s linear infinite;
}
@keyframes ticker { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
.ticker-item {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    margin-right: 40px;
    color: #6b7280;
}
.ticker-item .sym { color: #d1d5db; font-weight: 500; }
.ticker-item .pos { color: #34d399; }
.ticker-item .neg { color: #f87171; }

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1px;
    background: #1c2033;
    border: 1px solid #1c2033;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: #0d0f17;
    padding: 1.1rem 1.25rem;
    transition: background 0.15s;
}
.kpi-card:hover { background: #111420; }
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-sub {
    font-size: 11px;
    color: #4b5563;
    font-family: 'JetBrains Mono', monospace;
}
.green { color: #34d399; }
.red   { color: #f87171; }
.blue  { color: #60a5fa; }
.amber { color: #fbbf24; }
.gray  { color: #9ca3af; }

/* ── Section headers ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 0.9rem;
    padding-bottom: 8px;
    border-bottom: 1px solid #1c2033;
}
.sec-head-title {
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #9ca3af;
}
.sec-head-line {
    flex: 1;
    height: 1px;
    background: #1c2033;
}
.sec-head-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 3px;
    background: rgba(59,130,246,0.08);
    border: 1px solid #1c2033;
    color: #374151;
}

/* ── Chart container ── */
.chart-wrap {
    background: #0d0f17;
    border: 1px solid #1c2033;
    border-radius: 6px;
    padding: 1px;
    margin-bottom: 1rem;
}

/* ── Table ── */
.stDataFrame {
    border: 1px solid #1c2033 !important;
    border-radius: 6px !important;
}
.stDataFrame thead th {
    background: #111420 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #6b7280 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.stDataFrame tbody td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #9ca3af !important;
    border-color: #1c2033 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0f17;
    border-bottom: 1px solid #1c2033;
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500;
    color: #4b5563 !important;
    padding: 10px 22px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #f9fafb !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}

/* ── Button ── */
.stButton > button {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.6rem 1.25rem !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #2563eb !important; }

/* ── Progress ── */
.stProgress > div > div {
    background: #3b82f6 !important;
    border-radius: 2px !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0d0f17;
    border: 1px solid #1c2033;
    border-radius: 6px;
    padding: 0.75rem 1rem;
}
[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: #4b5563 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.15rem !important;
    color: #f9fafb !important;
}

/* ── Inputs ── */
.stTextInput input {
    background: #111420 !important;
    border: 1px solid #1c2033 !important;
    border-radius: 6px !important;
    color: #d1d5db !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
.stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

/* ── Info box ── */
.info-box {
    background: #0d0f17;
    border: 1px solid #1c2033;
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 1.25rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #6b7280;
    line-height: 2;
}

/* ── Divider ── */
hr { border-color: #1c2033 !important; }

/* ── Alerts ── */
.stAlert {
    background: #0d0f17 !important;
    border: 1px solid #1c2033 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Hide branding ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: #08090e; border-bottom: 1px solid #1c2033; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Plotly theme — applied individually, NO dict spreading conflict
# ─────────────────────────────────────────────────────────────
BG       = "#08090e"
PANEL_BG = "#0d0f17"
GRID     = "rgba(255,255,255,0.04)"
TEXT     = "#6b7280"
BORDER   = "#1c2033"
C_BLUE   = "#3b82f6"
C_GREEN  = "#34d399"
C_RED    = "#f87171"
C_AMBER  = "#fbbf24"
C_PURPLE = "#a78bfa"
C_CYAN   = "#22d3ee"

def style_fig(fig, height=320, xtitle="", ytitle="",
              xfmt="", yfmt="", show_legend=False):
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        height=height,
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=PANEL_BG,
        font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
        margin=dict(l=55, r=20, t=25, b=45),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TEXT),
            orientation="h", y=1.06, x=0,
        ),
    )
    fig.update_xaxes(
        title_text=xtitle, title_font=dict(size=11, color=TEXT),
        tickformat=xfmt, gridcolor=GRID,
        linecolor=BORDER, tickcolor=BORDER,
        tickfont=dict(color=TEXT, size=10),
    )
    fig.update_yaxes(
        title_text=ytitle, title_font=dict(size=11, color=TEXT),
        tickformat=yfmt, gridcolor=GRID,
        linecolor=BORDER, tickcolor=BORDER,
        tickfont=dict(color=TEXT, size=10),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">RISK<span>DESK</span></div>
    <span class="topbar-tag">PORTFOLIO ANALYTICS</span>
    <span class="topbar-tag">LIVE</span>
  </div>
  <div class="topbar-right">NIRMIT PATEL &nbsp;·&nbsp; MBA CANDIDATE &nbsp;·&nbsp; PACE UNIVERSITY</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Portfolio Config")
    st.markdown("---")

    ticker_input = st.text_input(
        "TICKERS",
        value="AAPL, MSFT, JPM, XOM, GLD",
        help="Any Yahoo Finance ticker, comma-separated",
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("FROM", value=pd.to_datetime("2019-01-01"))
    with col2:
        end_date = st.date_input("TO", value=pd.to_datetime("2024-12-31"))

    st.markdown("**WEIGHTS**")
    default_w = round(1.0 / len(tickers), 2)
    raw_w = [
        st.slider(t, 0.0, 1.0, default_w, 0.01, key=f"w_{t}")
        for t in tickers
    ]
    total = sum(raw_w) or 1
    weights = [w / total for w in raw_w]
    for t, w in zip(tickers, weights):
        st.progress(w, text=f"`{t}` — {w:.1%}")

    st.markdown("---")
    benchmark = st.selectbox("BENCHMARK", ["SPY", "QQQ", "DIA", "IWM"])
    rf_rate   = st.slider("RISK-FREE RATE %", 0.0, 8.0, 5.0, 0.25) / 100
    conf      = st.selectbox("VAR CONFIDENCE", [0.90, 0.95, 0.99], index=1)
    n_mc      = st.slider("MC PATHS", 500, 5000, 1000, 500)

    st.markdown("---")
    run = st.button("▶  RUN ANALYSIS")

# ─────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────
if not run:
    # Ticker tape
    st.markdown("""
    <div class="ticker-tape">
      <div class="ticker-inner">
        <span class="ticker-item"><span class="sym">AAPL</span> &nbsp;189.43 &nbsp;<span class="pos">+1.24%</span></span>
        <span class="ticker-item"><span class="sym">MSFT</span> &nbsp;415.22 &nbsp;<span class="pos">+0.87%</span></span>
        <span class="ticker-item"><span class="sym">JPM</span> &nbsp;201.11 &nbsp;<span class="neg">-0.32%</span></span>
        <span class="ticker-item"><span class="sym">GLD</span> &nbsp;187.90 &nbsp;<span class="pos">+0.61%</span></span>
        <span class="ticker-item"><span class="sym">SPY</span> &nbsp;521.34 &nbsp;<span class="pos">+0.44%</span></span>
        <span class="ticker-item"><span class="sym">NVDA</span> &nbsp;875.40 &nbsp;<span class="pos">+2.11%</span></span>
        <span class="ticker-item"><span class="sym">TSLA</span> &nbsp;177.90 &nbsp;<span class="neg">-1.05%</span></span>
        <span class="ticker-item"><span class="sym">XOM</span> &nbsp;112.33 &nbsp;<span class="pos">+0.19%</span></span>
        <span class="ticker-item"><span class="sym">QQQ</span> &nbsp;448.90 &nbsp;<span class="pos">+0.93%</span></span>
        <span class="ticker-item"><span class="sym">BLK</span> &nbsp;803.10 &nbsp;<span class="neg">-0.27%</span></span>
        <span class="ticker-item"><span class="sym">AAPL</span> &nbsp;189.43 &nbsp;<span class="pos">+1.24%</span></span>
        <span class="ticker-item"><span class="sym">MSFT</span> &nbsp;415.22 &nbsp;<span class="pos">+0.87%</span></span>
        <span class="ticker-item"><span class="sym">JPM</span> &nbsp;201.11 &nbsp;<span class="neg">-0.32%</span></span>
        <span class="ticker-item"><span class="sym">GLD</span> &nbsp;187.90 &nbsp;<span class="pos">+0.61%</span></span>
        <span class="ticker-item"><span class="sym">SPY</span> &nbsp;521.34 &nbsp;<span class="pos">+0.44%</span></span>
        <span class="ticker-item"><span class="sym">NVDA</span> &nbsp;875.40 &nbsp;<span class="pos">+2.11%</span></span>
        <span class="ticker-item"><span class="sym">TSLA</span> &nbsp;177.90 &nbsp;<span class="neg">-1.05%</span></span>
        <span class="ticker-item"><span class="sym">XOM</span> &nbsp;112.33 &nbsp;<span class="pos">+0.19%</span></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:680px;margin:3rem auto;">
      <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
           color:#f9fafb;line-height:1.1;margin-bottom:1rem;">
        Institutional-grade<br>
        <span style="color:#3b82f6;">portfolio risk analysis</span>
      </div>
      <div style="font-size:14px;color:#6b7280;line-height:1.8;margin-bottom:2rem;">
        Compute VaR, CVaR, Sharpe, Sortino, drawdown, beta, and alpha.
        Visualize the Markowitz efficient frontier and run Monte Carlo simulations
        on any combination of stocks, ETFs, or indices.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, title, desc in [
        (c1, "Risk Metrics", "VaR · CVaR · Sharpe · Sortino · Drawdown · Beta"),
        (c2, "Optimization", "Markowitz efficient frontier · Max Sharpe · Min Variance"),
        (c3, "Simulation",   "Monte Carlo fan chart · Percentile bands · Loss probability"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0d0f17;border:1px solid #1c2033;border-radius:8px;
                 padding:1.25rem;height:100px;">
              <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:13px;
                   color:#f9fafb;margin-bottom:6px;">{title}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                   color:#4b5563;line-height:1.7;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:2.5rem;font-family:'JetBrains Mono',monospace;
         font-size:12px;color:#374151;">
      Configure your portfolio in the sidebar → click  ▶ RUN ANALYSIS
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
with st.spinner("Fetching market data..."):
    loader = DataLoader()
    try:
        returns  = loader.get_returns(tickers, str(start_date), str(end_date))
        bench_r  = loader.get_benchmark(benchmark, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

bench_aligned = bench_r.reindex(returns.index).dropna()
returns       = returns.dropna()

if returns.empty:
    st.error("No return data found. Check your tickers and date range.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────────────────────
engine  = RiskEngine(returns, weights, rf_rate, bench_aligned)
metrics = engine.compute_all(conf)
c_int   = int(conf * 100)

ann_ret = metrics["ann_return"]
ann_vol = metrics["ann_vol"]
sharpe  = metrics["sharpe"]
sortino = metrics["sortino"]
mdd     = metrics["max_drawdown"]
beta    = metrics.get("beta") or 0.0

# ─────────────────────────────────────────────────────────────
# Ticker tape
# ─────────────────────────────────────────────────────────────
ticker_items = ""
for t, w in zip(tickers, weights):
    ticker_items += f'<span class="ticker-item"><span class="sym">{t}</span> &nbsp;{w:.1%}</span>'
ticker_items_2x = ticker_items * 2
st.markdown(f"""
<div class="ticker-tape">
  <div class="ticker-inner">{ticker_items_2x}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────
def color_class(v, positive_good=True):
    if positive_good:
        return "green" if v >= 0 else "red"
    return "red" if v < 0 else "green"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Ann. Return</div>
    <div class="kpi-value {color_class(ann_ret)}">{ann_ret:.1%}</div>
    <div class="kpi-sub">vs {benchmark}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Volatility</div>
    <div class="kpi-value blue">{ann_vol:.1%}</div>
    <div class="kpi-sub">annualized std dev</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Sharpe Ratio</div>
    <div class="kpi-value {'green' if sharpe >= 1 else 'amber' if sharpe >= 0.5 else 'red'}">{sharpe:.2f}</div>
    <div class="kpi-sub">{'excellent' if sharpe>=2 else 'good' if sharpe>=1 else 'average' if sharpe>=0.5 else 'below avg'}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Sortino Ratio</div>
    <div class="kpi-value {'green' if sortino >= 1 else 'amber' if sortino >= 0.5 else 'red'}">{sortino:.2f}</div>
    <div class="kpi-sub">downside-adjusted</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value red">{mdd:.1%}</div>
    <div class="kpi-sub">peak-to-trough</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Beta / {benchmark}</div>
    <div class="kpi-value blue">{beta:.2f}</div>
    <div class="kpi-sub">{'defensive' if beta < 0.8 else 'neutral' if beta < 1.2 else 'aggressive'}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Risk Metrics",
    "🎯  Efficient Frontier",
    "🔮  Monte Carlo",
    "🌡️  Correlations",
])

# ══════════ TAB 1 ══════════════════════════════════════════════
with tab1:
    cl, cr = st.columns(2, gap="large")

    with cl:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-title">Cumulative Returns</div>
          <div class="sec-head-line"></div>
          <div class="sec-head-badge">vs benchmark</div>
        </div>""", unsafe_allow_html=True)

        cum   = engine.cumulative_returns()
        bench = (1 + bench_aligned.reindex(cum.index).dropna()).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines",
            name="Portfolio", line=dict(color=C_BLUE, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=bench.index, y=bench.values, mode="lines",
            name=benchmark, line=dict(color=BORDER, width=1.5, dash="dot"),
        ))
        style_fig(fig, height=300, yfmt=".2f", show_legend=True)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-title">Portfolio Drawdown</div>
          <div class="sec-head-line"></div>
          <div class="sec-head-badge">peak-to-trough</div>
        </div>""", unsafe_allow_html=True)

        dd = engine.drawdown_series()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100, mode="lines",
            fill="tozeroy", line=dict(color=C_RED, width=1.5),
            fillcolor="rgba(248,113,113,0.08)",
        ))
        style_fig(fig2, height=300, yfmt=".1f")
        fig2.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""<div class="sec-head">
      <div class="sec-head-title">Rolling Sharpe Ratio</div>
      <div class="sec-head-line"></div>
      <div class="sec-head-badge">63-day window</div>
    </div>""", unsafe_allow_html=True)

    roll = engine.rolling_sharpe(63).dropna()
    fig3 = go.Figure()
    fig3.add_hrect(y0=1, y1=float(roll.max()) + 0.5,
                   fillcolor="rgba(52,211,153,0.03)", line_width=0)
    fig3.add_hline(y=0, line_dash="dot",
                   line_color="rgba(248,113,113,0.4)", line_width=1)
    fig3.add_hline(y=1, line_dash="dash",
                   line_color="rgba(52,211,153,0.3)", line_width=1)
    fig3.add_trace(go.Scatter(
        x=roll.index, y=roll.values, mode="lines",
        fill="tozeroy", line=dict(color=C_GREEN, width=1.5),
        fillcolor="rgba(52,211,153,0.06)",
    ))
    style_fig(fig3, height=220)
    st.plotly_chart(fig3, use_container_width=True)

    ca, cb = st.columns(2, gap="large")

    with ca:
        st.markdown(f"""<div class="sec-head">
          <div class="sec-head-title">VaR Comparison</div>
          <div class="sec-head-line"></div>
          <div class="sec-head-badge">{c_int}% confidence</div>
        </div>""", unsafe_allow_html=True)

        var_df = pd.DataFrame({
            "Method":      ["Historical",     "Parametric",      "Monte Carlo",   "CVaR / ES"],
            "1-Day Loss":  [
                f"{metrics[f'var_hist_{c_int}']:.2%}",
                f"{metrics[f'var_para_{c_int}']:.2%}",
                f"{metrics[f'var_mc_{c_int}']:.2%}",
                f"{metrics[f'cvar_{c_int}']:.2%}",
            ],
            "Assumption":  [
                "Empirical distribution",
                "Normal distribution",
                "Correlated GBM",
                "Avg loss in tail",
            ],
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)

    with cb:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-title">Full Risk Report</div>
          <div class="sec-head-line"></div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(engine.summary_df(conf), use_container_width=True)

# ══════════ TAB 2 ══════════════════════════════════════════════
with tab2:
    st.markdown("""<div class="sec-head">
      <div class="sec-head-title">Markowitz Efficient Frontier</div>
      <div class="sec-head-line"></div>
      <div class="sec-head-badge">3,000 random portfolios</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Optimizing..."):
        opt = PortfolioOptimizer(returns, rf_rate)
        rdf = opt.random_portfolios(3000)
        fdf = opt.efficient_frontier(60)
        ms  = opt.maximize_sharpe()
        mv  = opt.minimize_variance()

    w_arr  = np.array(weights)
    mu_ann = returns.mean().values * 252
    cv_ann = returns.cov().values * 252
    c_ret  = float(np.dot(w_arr, mu_ann))
    c_vol  = float(np.sqrt(w_arr @ cv_ann @ w_arr))
    c_sh   = (c_ret - rf_rate) / c_vol if c_vol > 0 else 0

    fig_f = go.Figure()

    # Random portfolios scatter
    fig_f.add_trace(go.Scatter(
        x=rdf["volatility"],
        y=rdf["return"],
        mode="markers",
        marker=dict(
            size=4,
            color=rdf["sharpe"],
            colorscale=[
                [0.0, "#0d0f17"],
                [0.3, "#1e3a5f"],
                [0.7, "#1d4ed8"],
                [1.0, "#60a5fa"],
            ],
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Sharpe",
                    font=dict(size=11, color=TEXT,
                              family="JetBrains Mono, monospace"),
                ),
                thickness=10,
                len=0.6,
                tickfont=dict(size=10, color=TEXT,
                              family="JetBrains Mono, monospace"),
                bgcolor=PANEL_BG,
                bordercolor=BORDER,
                borderwidth=1,
            ),
        ),
        name="Random portfolios",
        hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
    ))

    # Efficient frontier line
    if not fdf.empty:
        fig_f.add_trace(go.Scatter(
            x=fdf["volatility"], y=fdf["return"],
            mode="lines", name="Efficient frontier",
            line=dict(color=C_BLUE, width=2.5),
        ))

    # Max Sharpe
    fig_f.add_trace(go.Scatter(
        x=[ms["volatility"]], y=[ms["return"]],
        mode="markers+text",
        marker=dict(symbol="star", size=20, color=C_GREEN,
                    line=dict(width=1, color="white")),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(size=11, color=C_GREEN,
                      family="JetBrains Mono, monospace"),
        name=f"Max Sharpe ({ms['sharpe']:.2f})",
    ))

    # Min Variance
    fig_f.add_trace(go.Scatter(
        x=[mv["volatility"]], y=[mv["return"]],
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color=C_AMBER,
                    line=dict(width=1, color="white")),
        text=["Min Var"], textposition="top left",
        textfont=dict(size=11, color=C_AMBER,
                      family="JetBrains Mono, monospace"),
        name="Min Variance",
    ))

    # Current portfolio
    fig_f.add_trace(go.Scatter(
        x=[c_vol], y=[c_ret],
        mode="markers+text",
        marker=dict(symbol="circle", size=16, color=C_RED,
                    line=dict(width=2, color="white")),
        text=["Your Portfolio"], textposition="bottom right",
        textfont=dict(size=11, color=C_RED,
                      family="JetBrains Mono, monospace"),
        name=f"Your portfolio (Sharpe {c_sh:.2f})",
    ))

    style_fig(fig_f, height=500,
              xtitle="Annualized Volatility", xfmt=".0%",
              ytitle="Annualized Return",     yfmt=".0%",
              show_legend=True)
    fig_f.update_layout(
        legend=dict(
            orientation="v",
            yanchor="bottom", y=0.02,
            xanchor="right",  x=0.99,
            bgcolor="rgba(13,15,23,0.95)",
            bordercolor=BORDER, borderwidth=1,
        )
    )
    st.plotly_chart(fig_f, use_container_width=True)

    cm1, cm2, cm3 = st.columns(3, gap="large")
    for col, label, d in [
        (cm1, "Max Sharpe Portfolio",
         {"return": ms["return"], "volatility": ms["volatility"], "sharpe": ms["sharpe"]}),
        (cm2, "Min Variance Portfolio",
         {"return": mv["return"], "volatility": mv["volatility"], "sharpe": mv["sharpe"]}),
        (cm3, "Your Portfolio",
         {"return": c_ret, "volatility": c_vol, "sharpe": c_sh}),
    ]:
        with col:
            st.markdown(f"""<div class="sec-head">
              <div class="sec-head-title">{label}</div>
              <div class="sec-head-line"></div>
            </div>""", unsafe_allow_html=True)
            st.metric("Return",     f"{d['return']:.2%}")
            st.metric("Volatility", f"{d['volatility']:.2%}")
            st.metric("Sharpe",     f"{d['sharpe']:.2f}")

    st.markdown("""<div class="sec-head">
      <div class="sec-head-title">Weight Comparison</div>
      <div class="sec-head-line"></div>
    </div>""", unsafe_allow_html=True)
    st.dataframe(opt.weights_df(current_weights=weights),
                 use_container_width=True)

# ══════════ TAB 3 ══════════════════════════════════════════════
with tab3:
    st.markdown(f"""<div class="sec-head">
      <div class="sec-head-title">Monte Carlo Simulation</div>
      <div class="sec-head-line"></div>
      <div class="sec-head-badge">{n_mc:,} paths · 252 days · $10,000 start</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running simulation..."):
        sim   = MonteCarloSimulator(returns, weights)
        paths = sim.run(n=n_mc, horizon=252, start_value=10_000)

    days = np.arange(len(paths))
    p5   = paths.quantile(0.05, axis=1)
    p25  = paths.quantile(0.25, axis=1)
    p50  = paths.quantile(0.50, axis=1)
    p75  = paths.quantile(0.75, axis=1)
    p95  = paths.quantile(0.95, axis=1)

    fig_mc = go.Figure()

    # Outer band
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=pd.concat([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th pct",
    ))

    # Inner band
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=pd.concat([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th–75th pct",
    ))

    # Sample paths
    for i in range(min(60, paths.shape[1])):
        fig_mc.add_trace(go.Scatter(
            x=days, y=paths.iloc[:, i], mode="lines",
            line=dict(color="rgba(59,130,246,0.04)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Median
    fig_mc.add_trace(go.Scatter(
        x=days, y=p50, mode="lines",
        line=dict(color=C_BLUE, width=2.5),
        name="Median path",
        hovertemplate="Day %{x}<br><b>$%{y:,.0f}</b><extra></extra>",
    ))

    # Break-even line
    fig_mc.add_hline(y=10000, line_dash="dot",
                     line_color="rgba(248,113,113,0.35)", line_width=1.5,
                     annotation_text="Break-even",
                     annotation_font=dict(size=10, color=C_RED))

    style_fig(fig_mc, height=460,
              xtitle="Trading Days",
              ytitle="Portfolio Value ($)",
              show_legend=True)
    fig_mc.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_mc, use_container_width=True)

    final = paths.iloc[-1]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Median Outcome",  f"${final.median():,.0f}")
    mc2.metric("5th Percentile",  f"${final.quantile(0.05):,.0f}")
    mc3.metric("95th Percentile", f"${final.quantile(0.95):,.0f}")
    mc4.metric("Prob. of Loss",   f"{(final < 10000).mean():.1%}")

# ══════════ TAB 4 ══════════════════════════════════════════════
with tab4:
    ch, cd = st.columns([1.1, 1], gap="large")

    with ch:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-title">Correlation Matrix</div>
          <div class="sec-head-line"></div>
          <div class="sec-head-badge">Pearson · daily returns</div>
        </div>""", unsafe_allow_html=True)

        corr = engine.correlation_matrix()
        fig_h = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=[
                [0.0, "#0d0f17"],
                [0.25, "#1e3a5f"],
                [0.5, "#1d4ed8"],
                [0.75, "#3b82f6"],
                [1.0, "#93c5fd"],
            ],
            zmin=-1,
            zmax=1,
        )
        fig_h.update_traces(
            textfont=dict(size=13, color="#d1d5db",
                          family="JetBrains Mono, monospace"),
        )
        fig_h.update_layout(
            height=400,
            plot_bgcolor=PANEL_BG,
            paper_bgcolor=PANEL_BG,
            font=dict(family="JetBrains Mono, monospace",
                      color=TEXT, size=11),
            margin=dict(l=20, r=60, t=20, b=20),
            coloraxis_colorbar=dict(
                title=dict(
                    text="Corr",
                    font=dict(size=11, color=TEXT,
                              family="JetBrains Mono, monospace"),
                ),
                thickness=10,
                len=0.8,
                tickfont=dict(size=10, color=TEXT,
                              family="JetBrains Mono, monospace"),
                bgcolor=PANEL_BG,
                bordercolor=BORDER,
                borderwidth=1,
            ),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with cd:
        st.markdown("""<div class="sec-head">
          <div class="sec-head-title">Return Distributions</div>
          <div class="sec-head-line"></div>
          <div class="sec-head-badge">violin + box</div>
        </div>""", unsafe_allow_html=True)

        palette = [C_BLUE, C_GREEN, C_AMBER, C_RED, C_PURPLE, C_CYAN]
        fig_v = go.Figure()
        for i, ticker in enumerate(returns.columns):
            clr = palette[i % len(palette)]
            r = int(clr[1:3], 16)
            g = int(clr[3:5], 16)
            b = int(clr[5:7], 16)
            fig_v.add_trace(go.Violin(
                y=returns[ticker] * 100,
                name=ticker,
                box_visible=True,
                meanline_visible=True,
                fillcolor=f"rgba({r},{g},{b},0.12)",
                line_color=clr,
                meanline=dict(color=clr, width=2),
            ))
        style_fig(fig_v, height=400, show_legend=False)
        fig_v.update_yaxes(ticksuffix="%", title_text="Daily Return (%)")
        st.plotly_chart(fig_v, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.25rem 0;
     border-top:1px solid #1c2033;
     display:flex;justify-content:space-between;align-items:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#374151;">
    Built by &nbsp;<span style="color:#9ca3af;font-weight:500;">Nirmit Patel</span>
    &nbsp;·&nbsp; MBA Candidate, Business Analytics &nbsp;·&nbsp; Pace University Lubin School
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;display:flex;gap:16px;">
    <a href="https://github.com/Nirmitpatel889" target="_blank"
       style="color:#3b82f6;text-decoration:none;">GitHub</a>
    <a href="https://nirmit-patel-portfolio.netlify.app" target="_blank"
       style="color:#3b82f6;text-decoration:none;">Portfolio</a>
  </div>
</div>
""", unsafe_allow_html=True)
