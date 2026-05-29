"""
Sub-Sector Index Performance Hub
=================================
A Streamlit application that analyzes US sub-sector ETF performance
across multiple time horizons using Yahoo Finance data.

Author: Arjun Garg
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Sub-Sector Index Hub",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK       = "SPY"
BENCHMARK_LABEL = "S&P 500 (SPY)"
RF_ANNUAL       = 0.0425          # 4.25% risk-free rate
CSV_PATH        = "us_subsector_etfs.csv"

TIMEFRAME_MAP = {
    "6 Years":   72,
    "5 Years":   60,
    "4 Years":   48,
    "3 Years":   36,
    "2 Years":   24,
    "1 Year":    12,
    "6 Months":   6,
    "3 Months":   3,
}

# Color palette for the top-6 cards and chart lines
TOP6_COLORS = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899", "#06b6d4"]
SPY_COLOR   = "#ef4444"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── page background ── */
.stApp {
    background: #020617;
    color: #f1f5f9;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p {
    color: #94a3b8 !important;
    font-size: 13px;
}

/* ── metric cards ── */
.card-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.circle-card {
    width: 148px;
    height: 148px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 12px;
    margin: auto;
    position: relative;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.circle-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 28px rgba(59,130,246,0.4);
}
.circle-card-spy:hover {
    box-shadow: 0 0 28px rgba(239,68,68,0.4);
}
.circle-ticker {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 15px;
    color: #f1f5f9;
    letter-spacing: 0.05em;
}
.circle-name {
    font-size: 9.5px;
    color: #94a3b8;
    margin-top: 1px;
    max-width: 110px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.circle-return {
    font-size: 17px;
    font-weight: 700;
    margin-top: 4px;
}
.circle-return-positive { color: #10b981; }
.circle-return-negative { color: #ef4444; }
.circle-metrics {
    font-size: 10px;
    color: #cbd5e1;
    margin-top: 3px;
    line-height: 1.55;
}
.rank-badge {
    font-size: 10px;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── section headers ── */
.section-header {
    font-size: 18px;
    font-weight: 600;
    color: #f1f5f9;
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}
.section-header-spy {
    border-left-color: #ef4444;
}

/* ── stat pills ── */
.stat-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 500;
    margin: 2px;
}

/* ── divider ── */
hr { border-color: #1e293b !important; }

/* ── table styling ── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* ── inputs ── */
.stTextInput > div > div > input {
    background: #0f172a;
    border: 1px solid #1e293b;
    color: #f1f5f9;
    border-radius: 8px;
    font-size: 13px;
}
.stSelectbox > div > div {
    background: #0f172a;
    border: 1px solid #1e293b;
    color: #f1f5f9;
    border-radius: 8px;
}
.stMultiSelect > div > div {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
}

/* ── buttons ── */
.stButton > button {
    background: #1e40af;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    padding: 6px 16px;
    font-family: 'Inter', sans-serif;
}
.stButton > button:hover {
    background: #2563eb;
}

/* ── info box ── */
.info-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.7;
}

/* ── top strip ── */
.top-strip {
    background: linear-gradient(90deg, #020617, #0f172a, #020617);
    border-bottom: 1px solid #1e293b;
    padding: 6px 0;
    font-size: 12px;
    color: #475569;
    text-align: right;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_metadata() -> pd.DataFrame:
    """Load and clean the ETF metadata CSV."""
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df["Ticker"]     = df["Ticker"].str.strip().str.upper()
    df["Sub_Sector"] = df["Sub_Sector"].str.strip()
    df["Category"]   = df["Category"].str.strip()
    df["Issuer"]     = df["Issuer"].str.strip()
    # Drop rows with blank tickers
    df = df.dropna(subset=["Ticker"])
    df = df[df["Ticker"] != ""]
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list[str], months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices from Yahoo Finance,
    resample to monthly, normalize, and compute performance metrics.

    Returns:
        normalized_prices : DataFrame with $1-normalized monthly prices
        metrics_df        : DataFrame with return/risk metrics per ticker
    """
    end_date   = datetime.today()
    start_date = end_date - relativedelta(months=months + 3)   # extra buffer for resampling

    # Always include benchmark
    all_tickers = sorted(set(tickers + [BENCHMARK]))

    try:
        raw = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        st.error(f"Yahoo Finance download failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Extract close prices — handle single vs multi ticker download
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers

    # Drop completely empty columns
    prices.dropna(axis=1, how="all", inplace=True)

    # Trim to the exact window requested
    window_start = end_date - relativedelta(months=months)
    prices = prices[prices.index >= pd.Timestamp(window_start)]

    # Resample to month-end
    monthly = prices.resample("ME").last().ffill()

    # Drop tickers that still have all-NaN after resample
    monthly.dropna(axis=1, how="all", inplace=True)

    if monthly.empty or len(monthly) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Normalize each column so first value = $1
    normalized = monthly.div(monthly.iloc[0])

    # ── compute metrics ──────────────────────────────────────────────────────
    rf_monthly = RF_ANNUAL / 12
    rows = []

    for ticker in normalized.columns:
        series       = normalized[ticker].dropna()
        if len(series) < 2:
            continue

        monthly_rets = series.pct_change().dropna()
        if monthly_rets.empty:
            continue

        total_ret    = float(series.iloc[-1]) - 1.0
        years        = max(len(series) - 1, 1) / 12
        ann_ret      = (1 + total_ret) ** (1 / years) - 1

        excess_rets  = monthly_rets - rf_monthly
        volatility   = monthly_rets.std() * np.sqrt(12)
        sharpe       = (ann_ret - RF_ANNUAL) / volatility if volatility > 1e-6 else 0.0

        downside     = monthly_rets[monthly_rets < rf_monthly]
        down_vol     = downside.std() * np.sqrt(12) if len(downside) > 1 else 0.0
        sortino      = (ann_ret - RF_ANNUAL) / down_vol if down_vol > 1e-6 else 0.0

        rows.append({
            "Ticker":           ticker,
            "Total Return %":   round(total_ret   * 100, 2),
            "Annual Return %":  round(ann_ret      * 100, 2),
            "Sharpe Ratio":     round(sharpe,             2),
            "Sortino Ratio":    round(sortino,            2),
            "$1 Growth":        round(float(series.iloc[-1]), 3),
        })

    metrics = pd.DataFrame(rows)
    return normalized, metrics


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: CIRCLE CARD HTML
# ─────────────────────────────────────────────────────────────────────────────

def make_circle_card(
    ticker:       str,
    sub_sector:   str,
    ann_ret:      float,
    sharpe:       float,
    sortino:      float,
    dollar_val:   float,
    border_color: str,
    bg_gradient:  str,
    rank:         str = "",
) -> str:
    ret_class = "circle-return-positive" if ann_ret >= 0 else "circle-return-negative"
    sign      = "+" if ann_ret >= 0 else ""
    name_short = sub_sector[:18] + "…" if len(sub_sector) > 18 else sub_sector

    badge_html = f'<div class="rank-badge">{rank}</div>' if rank else ""

    return f"""
    <div class="card-wrapper">
        {badge_html}
        <div class="circle-card" style="
            background: {bg_gradient};
            border: 2.5px solid {border_color};
            box-shadow: 0 0 18px {border_color}33;
        ">
            <div class="circle-ticker">{ticker}</div>
            <div class="circle-name" title="{sub_sector}">{name_short}</div>
            <div class="circle-return {ret_class}">{sign}{ann_ret:.1f}%</div>
            <div class="circle-metrics">
                Sharpe: {sharpe:.2f}<br>
                Sortino: {sortino:.2f}<br>
                $1 → <b style="color:#f1f5f9">${dollar_val:.2f}</b>
            </div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: PLOTLY CHART
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(
    normalized:     pd.DataFrame,
    chart_tickers:  list[str],
    ticker_labels:  dict[str, str],
    color_map:      dict[str, str],
) -> go.Figure:
    """Build an elegant Plotly multi-line normalized return chart."""

    fig = go.Figure()

    for ticker in chart_tickers:
        if ticker not in normalized.columns:
            continue

        series = normalized[ticker].dropna()
        dates  = series.index.strftime("%Y-%m")
        color  = color_map.get(ticker, "#64748b")
        label  = ticker_labels.get(ticker, ticker)
        is_spy = ticker == BENCHMARK

        fig.add_trace(go.Scatter(
            x          = dates,
            y          = series.values,
            mode       = "lines",
            name       = label,
            line       = dict(
                color = color,
                width = 2.5 if not is_spy else 1.8,
                dash  = "dot" if is_spy else "solid",
            ),
            hovertemplate = (
                f"<b>{label}</b><br>"
                "Date: %{x}<br>"
                "$1 → <b>$%{y:.3f}</b><extra></extra>"
            ),
        ))

    # $1 reference line
    fig.add_hline(
        y           = 1.0,
        line_dash   = "dash",
        line_color  = "#334155",
        line_width  = 1,
        annotation_text      = "$1.00 baseline",
        annotation_position  = "bottom left",
        annotation_font_color= "#475569",
        annotation_font_size = 10,
    )

    fig.update_layout(
        template        = "plotly_dark",
        paper_bgcolor   = "#0a0f1e",
        plot_bgcolor    = "#0a0f1e",
        font            = dict(family="Inter, sans-serif", color="#94a3b8", size=12),
        legend          = dict(
            bgcolor       = "#0f172a",
            bordercolor   = "#1e293b",
            borderwidth   = 1,
            font          = dict(size=11),
            orientation   = "h",
            yanchor       = "bottom",
            y             = 1.02,
            xanchor       = "left",
            x             = 0,
        ),
        hovermode       = "x unified",
        hoverlabel      = dict(
            bgcolor     = "#0f172a",
            bordercolor = "#1e293b",
            font_size   = 12,
            font_family = "JetBrains Mono",
        ),
        xaxis = dict(
            title         = "",
            showgrid      = True,
            gridcolor     = "#1e293b",
            gridwidth     = 0.5,
            tickangle     = -30,
            tickfont      = dict(size=10, color="#475569"),
            showline      = False,
        ),
        yaxis = dict(
            title         = "Value of $1 Invested",
            showgrid      = True,
            gridcolor     = "#1e293b",
            gridwidth     = 0.5,
            tickfont      = dict(size=10, color="#475569"),
            tickprefix    = "$",
            tickformat    = ".2f",
            showline      = False,
            zeroline      = False,
        ),
        margin          = dict(t=60, b=40, l=60, r=20),
        height          = 430,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(metadata: pd.DataFrame) -> tuple[int, list[str]]:
    """Render sidebar controls. Returns (months, extra_tickers)."""

    st.sidebar.markdown("""
    <div style="text-align:center; padding: 12px 0 20px 0;">
        <div style="font-size:20px; font-weight:700; color:#f1f5f9; letter-spacing:0.03em;">📊 Index Hub</div>
        <div style="font-size:11px; color:#475569; margin-top:4px;">Sub-Sector Performance</div>
    </div>
    """, unsafe_allow_html=True)

    # Timeframe
    st.sidebar.markdown("**Timeframe**")
    selected_tf = st.sidebar.selectbox(
        "Select Timeframe",
        list(TIMEFRAME_MAP.keys()),
        index=2,            # default: 3 Years
        label_visibility="collapsed",
    )
    months = TIMEFRAME_MAP[selected_tf]

    st.sidebar.markdown("---")

    # Custom tickers
    st.sidebar.markdown("**Add Custom Tickers**")
    st.sidebar.markdown(
        "<span style='font-size:11px;color:#475569;'>Comma-separated, e.g. AAPL, TSLA</span>",
        unsafe_allow_html=True,
    )
    custom_raw   = st.sidebar.text_input("Custom tickers", value="", label_visibility="collapsed")
    extra_tickers = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]

    st.sidebar.markdown("---")

    # About
    st.sidebar.markdown("""
    <div style="font-size:11px; color:#475569; line-height:1.7;">
        <b style="color:#64748b;">Data:</b> Yahoo Finance (adjusted close)<br>
        <b style="color:#64748b;">Granularity:</b> Monthly<br>
        <b style="color:#64748b;">Risk-free rate:</b> 4.25% annualized<br>
        <b style="color:#64748b;">Benchmark:</b> SPY (S&P 500 ETF)<br>
        <b style="color:#64748b;">Universe:</b> 184 sub-sector ETFs
    </div>
    """, unsafe_allow_html=True)

    return months, extra_tickers, selected_tf


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: HOME (Top 6 Cards + Chart)
# ─────────────────────────────────────────────────────────────────────────────

def render_home(
    normalized: pd.DataFrame,
    metrics_df: pd.DataFrame,
    metadata:   pd.DataFrame,
    months:     int,
    selected_tf: str,
    extra_tickers: list[str],
):
    # ── top-6 + SPY ──────────────────────────────────────────────────────────
    sector_metrics = metrics_df[metrics_df["Ticker"] != BENCHMARK].copy()
    top6           = sector_metrics.sort_values("Annual Return %", ascending=False).head(6)
    spy_row        = metrics_df[metrics_df["Ticker"] == BENCHMARK]

    if spy_row.empty:
        st.warning("Benchmark (SPY) data unavailable for this period.")
        return

    spy = spy_row.iloc[0]

    # Build sub-sector lookup
    meta_lookup = metadata.set_index("Ticker")["Sub_Sector"].to_dict()

    # ── header strip ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px;">
        <div>
            <div style="font-size:22px; font-weight:700; color:#f1f5f9;">
                🏆 Top 6 Performing Sectors
            </div>
            <div style="font-size:13px; color:#475569; margin-top:3px;">
                {selected_tf} lookback · Monthly granularity · vs S&P 500 benchmark
            </div>
        </div>
        <div style="font-size:11px; color:#334155; text-align:right; line-height:1.8;">
            RF Rate: 4.25%<br>Universe: 184 ETFs
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 7 circle cards ───────────────────────────────────────────────────────
    cols = st.columns(7, gap="small")

    for i, (_, row) in enumerate(top6.iterrows()):
        ticker     = row["Ticker"]
        sub_sector = meta_lookup.get(ticker, ticker)
        color      = TOP6_COLORS[i % len(TOP6_COLORS)]
        rank_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"]

        with cols[i]:
            st.markdown(make_circle_card(
                ticker       = ticker,
                sub_sector   = sub_sector,
                ann_ret      = row["Annual Return %"],
                sharpe       = row["Sharpe Ratio"],
                sortino      = row["Sortino Ratio"],
                dollar_val   = row["$1 Growth"],
                border_color = color,
                bg_gradient  = f"linear-gradient(145deg, #0f172a, #1e293b)",
                rank         = rank_labels[i],
            ), unsafe_allow_html=True)

    # SPY card
    with cols[6]:
        st.markdown(make_circle_card(
            ticker       = "SPY",
            sub_sector   = "S&P 500 Benchmark",
            ann_ret      = spy["Annual Return %"],
            sharpe       = spy["Sharpe Ratio"],
            sortino      = spy["Sortino Ratio"],
            dollar_val   = spy["$1 Growth"],
            border_color = SPY_COLOR,
            bg_gradient  = "linear-gradient(145deg, #1a0505, #2d0707)",
            rank         = "Benchmark",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── metrics comparison bar ────────────────────────────────────────────────
    st.markdown("---")
    mcols = st.columns(4)
    best  = top6.iloc[0] if not top6.empty else None

    if best is not None:
        with mcols[0]:
            delta = round(best["Annual Return %"] - spy["Annual Return %"], 1)
            st.metric(
                "Best Sector vs SPY",
                f"{best['Annual Return %']:.1f}%",
                delta=f"{delta:+.1f}% vs benchmark",
                delta_color="normal",
            )
        with mcols[1]:
            st.metric("Best Sharpe (Top 6)", f"{top6['Sharpe Ratio'].max():.2f}",
                      help="Higher = better risk-adjusted return")
        with mcols[2]:
            st.metric("Best Sortino (Top 6)", f"{top6['Sortino Ratio'].max():.2f}",
                      help="Higher = better downside-adjusted return")
        with mcols[3]:
            best_growth = top6["$1 Growth"].max()
            st.metric("Best $1 Growth", f"${best_growth:.2f}",
                      help=f"$1 invested {selected_tf.lower()} ago")

    st.markdown("---")

    # ── normalized growth chart ───────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📈 $1 Invested — Normalized Monthly Return</div>',
        unsafe_allow_html=True,
    )

    chart_tickers = top6["Ticker"].tolist() + [BENCHMARK] + extra_tickers
    chart_tickers = [t for t in chart_tickers if t in normalized.columns]

    color_map = {t: TOP6_COLORS[i] for i, t in enumerate(top6["Ticker"].tolist())}
    color_map[BENCHMARK] = SPY_COLOR
    for j, t in enumerate(extra_tickers):
        color_map[t] = px.colors.qualitative.Set2[j % len(px.colors.qualitative.Set2)]

    label_map = {t: f"{t} — {meta_lookup.get(t, t)}" for t in chart_tickers}
    label_map[BENCHMARK] = "SPY — S&P 500 ★"

    if chart_tickers:
        fig = build_chart(normalized, chart_tickers, label_map, color_map)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No chart data available for the selected tickers.")

    # ── extra tickers metrics (if any) ───────────────────────────────────────
    if extra_tickers:
        extra_data = metrics_df[metrics_df["Ticker"].isin(extra_tickers)]
        if not extra_data.empty:
            st.markdown("**Custom Ticker Metrics**")
            st.dataframe(
                extra_data.set_index("Ticker").style
                .format({
                    "Total Return %":  "{:+.2f}%",
                    "Annual Return %": "{:+.2f}%",
                    "Sharpe Ratio":    "{:.2f}",
                    "Sortino Ratio":   "{:.2f}",
                    "$1 Growth":       "${:.3f}",
                })
                .background_gradient(subset=["Annual Return %"], cmap="RdYlGn"),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: ALL METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def render_all_metrics(metrics_df: pd.DataFrame, metadata: pd.DataFrame):
    st.markdown(
        '<div class="section-header">📊 Full Performance Table — All Loaded ETFs</div>',
        unsafe_allow_html=True,
    )

    # Merge with metadata
    merged = metrics_df.merge(
        metadata[["Ticker", "Sub_Sector", "Category", "Issuer"]],
        on="Ticker",
        how="left",
    )
    merged = merged.sort_values("Annual Return %", ascending=False).reset_index(drop=True)
    merged.index += 1   # rank from 1

    # Sort control
    sort_col = st.selectbox(
        "Sort by",
        ["Annual Return %", "Total Return %", "Sharpe Ratio", "Sortino Ratio", "$1 Growth"],
        index=0,
        key="sort_all",
    )
    asc = st.checkbox("Ascending", value=False, key="asc_all")
    merged = merged.sort_values(sort_col, ascending=asc).reset_index(drop=True)
    merged.index += 1

    # Style
    styled = (
        merged.style
        .format({
            "Total Return %":  "{:+.2f}%",
            "Annual Return %": "{:+.2f}%",
            "Sharpe Ratio":    "{:.2f}",
            "Sortino Ratio":   "{:.2f}",
            "$1 Growth":       "${:.3f}",
        })
        .background_gradient(subset=["Annual Return %"], cmap="RdYlGn", vmin=-50, vmax=100)
        .background_gradient(subset=["Sharpe Ratio"],    cmap="Blues",  vmin=0,   vmax=3)
        .background_gradient(subset=["Sortino Ratio"],   cmap="Purples",vmin=0,   vmax=4)
    )

    st.dataframe(styled, use_container_width=True, height=520)

    # Download button
    csv_bytes = merged.to_csv(index=False).encode()
    st.download_button(
        label     = "⬇️ Download Results as CSV",
        data      = csv_bytes,
        file_name = "etf_performance.csv",
        mime      = "text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

def render_explorer(metadata: pd.DataFrame):
    st.markdown(
        '<div class="section-header">🔍 ETF Universe Explorer — All 184 Sub-Sector Indexes</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        search = st.text_input(
            "Search by ticker, sub-sector, category, or issuer",
            placeholder="e.g. Semiconductor, VanEck, Energy...",
            key="explorer_search",
        )
    with col2:
        categories = ["All Categories"] + sorted(metadata["Category"].dropna().unique().tolist())
        cat_filter = st.selectbox("Filter by Category", categories, key="cat_filter")
    with col3:
        issuers = ["All Issuers"] + sorted(metadata["Issuer"].dropna().unique().tolist())
        iss_filter = st.selectbox("Filter by Issuer", issuers, key="iss_filter")

    filtered = metadata.copy()

    if search:
        mask = (
            filtered["Ticker"].str.contains(search, case=False, na=False)
            | filtered["Sub_Sector"].str.contains(search, case=False, na=False)
            | filtered["Category"].str.contains(search, case=False, na=False)
            | filtered["Issuer"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    if cat_filter != "All Categories":
        filtered = filtered[filtered["Category"] == cat_filter]

    if iss_filter != "All Issuers":
        filtered = filtered[filtered["Issuer"] == iss_filter]

    # Summary counts
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Showing",    f"{len(filtered)} ETFs")
    info_col2.metric("Categories", filtered["Category"].nunique())
    info_col3.metric("Issuers",    filtered["Issuer"].nunique())
    info_col4.metric("Total Universe", f"{len(metadata)} ETFs")

    display = filtered[["Ticker", "Sub_Sector", "Category", "Issuer"]].reset_index(drop=True)
    display.index += 1

    st.dataframe(
        display,
        use_container_width=True,
        height=480,
        column_config={
            "Ticker":     st.column_config.TextColumn("Ticker",     width="small"),
            "Sub_Sector": st.column_config.TextColumn("Sub-Sector", width="medium"),
            "Category":   st.column_config.TextColumn("Category",   width="small"),
            "Issuer":     st.column_config.TextColumn("Issuer",     width="small"),
        },
    )

    # Category breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    cat_counts = filtered["Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]

    fig_cat = px.bar(
        cat_counts,
        x="Category",
        y="Count",
        color="Count",
        color_continuous_scale="Blues",
        title="ETF Count by Category (filtered)",
    )
    fig_cat.update_layout(
        template        = "plotly_dark",
        paper_bgcolor   = "#0a0f1e",
        plot_bgcolor    = "#0a0f1e",
        height          = 280,
        showlegend      = False,
        coloraxis_showscale = False,
        margin          = dict(t=40, b=30, l=40, r=20),
        xaxis_tickangle = -30,
        font            = dict(size=11, color="#94a3b8"),
        title_font      = dict(size=14, color="#94a3b8"),
    )
    st.plotly_chart(fig_cat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Sidebar
    metadata = load_metadata()
    months, extra_tickers, selected_tf = render_sidebar(metadata)

    # Page title
    st.markdown("""
    <div style="margin-bottom: 6px;">
        <span style="font-size:28px; font-weight:700; color:#f1f5f9;">
            Sub-Sector Index Performance Hub
        </span>
        <span style="font-size:13px; color:#475569; margin-left:12px;">
            Powered by Yahoo Finance
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Navigation tabs
    tab_home, tab_metrics, tab_explorer = st.tabs([
        "🏠  Dashboard",
        "📋  Full Metrics",
        "🔍  ETF Explorer",
    ])

    # ── fetch data ────────────────────────────────────────────────────────────
    all_tickers = metadata["Ticker"].dropna().tolist() + extra_tickers

    with st.spinner(f"Downloading market data for {len(all_tickers)} tickers…"):
        normalized, metrics_df = fetch_prices(all_tickers, months)

    if metrics_df.empty:
        st.error(
            "No market data could be retrieved. "
            "Check your internet connection or try a shorter timeframe — "
            "some ETFs may not have data going back that far."
        )
        st.stop()

    # ── render tabs ───────────────────────────────────────────────────────────
    with tab_home:
        render_home(normalized, metrics_df, metadata, months, selected_tf, extra_tickers)

    with tab_metrics:
        render_all_metrics(metrics_df, metadata)

    with tab_explorer:
        render_explorer(metadata)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#334155; text-align:center; padding:8px 0 16px 0; line-height:1.9;">
        Data via Yahoo Finance (adjusted close prices) · Risk-free rate 4.25% annualized ·
        Sharpe & Sortino ratios annualized from monthly returns · <b>Not financial advice</b>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
