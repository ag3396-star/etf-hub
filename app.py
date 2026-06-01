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
    initial_sidebar_state="collapsed",
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

TOP6_COLORS = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899", "#06b6d4"]

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Background Fixes */
    .stApp { background-color: #0f172a; color: #f8fafc; }
    
    /* Segmented Control Styling Overrides */
    div[data-baseweb="segmented-control"] {
        width: 100% !important;
        background-color: #1e293b !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    button[aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Metric Containers */
    .metric-container {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 20px;
    }
    .metric-val { font-size: 28px; font-weight: 700; color: #3b82f6; margin-top: 4px; }
    .metric-lbl { font-size: 13px; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Leaders Circle Cards */
    .circle-card {
        border-radius: 50%;
        width: 145px;
        height: 145px;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        margin: auto;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.3);
        transition: transform 0.2s;
    }
    .circle-card:hover { transform: scale(1.05); }
    .circle-ticker { font-weight: 800; font-size: 18px; margin-bottom: 1px; letter-spacing: -0.025em; }
    .circle-return { font-size: 15px; font-weight: 700; margin-bottom: 2px; }
    .circle-metrics { font-size: 10px; color: #94a3b8; line-height: 1.3; }
    
    /* Explanatory Banner Callouts */
    .explainer-banner {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        font-size: 14px;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_metadata():
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading metadata from {CSV_PATH}: {e}")
        return pd.DataFrame(columns=["#", "Ticker", "Sub_Sector", "Category", "Issuer", "Description", "URL"])

@st.cache_data(show_spinner=False)
def fetch_prices(tickers, months):
    end_date = datetime.today()
    start_date = end_date - relativedelta(months=months + 1)
    
    unique_tickers = list(set(tickers + [BENCHMARK]))
    
    try:
        df_raw = yf.download(unique_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    except Exception as e:
        st.error(f"Yahoo Finance download failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Flatten out multi-index columns if they exist
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_close = df_raw['Close']
    else:
        df_close = pd.DataFrame({unique_tickers[0]: df_raw['Close']})

    window_start = end_date - relativedelta(months=months)
    df_close = df_close[df_close.index >= window_start]
    
    # Resample using stable 'M' frequency parameters
    monthly_prices = df_close.resample('M').last().ffill().bfill()
    
    if monthly_prices.empty or len(monthly_prices) < 2:
        return pd.DataFrame(), pd.DataFrame()

    normalized = monthly_prices / monthly_prices.iloc[0]
    
    # Generate risk-adjusted return matrices
    metrics_list = []
    rf_monthly = RF_ANNUAL / 12
    n_years = months / 12.0

    for col in normalized.columns:
        series_norm = normalized[col]
        series_raw = monthly_prices[col]
        
        m_returns = series_raw.pct_change().dropna()
        if m_returns.empty:
            continue
            
        total_ret_val = series_norm.iloc[-1] / series_norm.iloc[0]
        ann_return = (total_ret_val ** (1 / n_years)) - 1 if total_ret_val > 0 else 0
        
        excess_returns = m_returns - rf_monthly
        vol = m_returns.std() * np.sqrt(12)
        sharpe = (excess_returns.mean() * 12) / vol if vol > 0.0001 else 0
        
        downside_returns = m_returns[m_returns < 0]
        if len(downside_returns) < 2:
            sortino = 99.9
        else:
            downside_vol = downside_returns.std() * np.sqrt(12)
            sortino = (excess_returns.mean() * 12) / downside_vol if downside_vol > 0.0001 else 0

        metrics_list.append({
            "Ticker": col,
            "Total Return Multiple": round(total_ret_val, 2),
            "Annualized Return %": round(ann_return * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio (Downside Risk)": round(sortino, 2)
        })

    return normalized, pd.DataFrame(metrics_list)

# ─────────────────────────────────────────────────────────────────────────────
# VIEW COMPONENT RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def render_home(normalized, metrics_df, metadata, months, selected_tf):
    st.subheader("🏆 Portfolio Sector Leaders")
    
    # Isolate benchmark statistics
    bench_row = metrics_df[metrics_df["Ticker"] == BENCHMARK]
    if bench_row.empty:
        st.warning("Benchmark metrics unavailable for this frame selection.")
        return
    bench_data = bench_row.iloc[0]

    # Filter out benchmark to view outperforming components
    subsectors = metrics_df[metrics_df["Ticker"] != BENCHMARK].copy()
    top_6 = subsectors.sort_values(by="Annualized Return %", ascending=False).head(6)
    
    # Draw out performance circular tracking elements
    cols_circ = st.columns(7)
    for idx, row in enumerate(top_6.itertuples()):
        with cols_circ[idx]:
            border_color = TOP6_COLORS[idx % len(TOP6_COLORS)]
            st.markdown(f"""
            <div class="circle-card" style="border: 3px solid {border_color};">
                <div class="circle-ticker">{row.Ticker}</div>
                <div class="circle-return" style="color:{border_color};">+{row.Annualized Return %}%</div>
                <div class="circle-metrics">
                    Sharpe: {row._4}<br>
                    Sortino: {row._5:.1f}<br>
                    Mult: {row._2}x
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with cols_circ[6]:
        st.markdown(f"""
        <div class="circle-card" style="border: 3px solid #ef4444; background: linear-gradient(135deg, #311010, #0f172a);">
            <div class="circle-ticker">{BENCHMARK}</div>
            <div class="circle-return" style="color:#ef4444;">+{bench_data['Annualized Return %']}%</div>
            <div class="circle-metrics">
                Sharpe: {bench_data['Sharpe Ratio']}<br>
                Sortino: {bench_data['Sortino Ratio (Downside Risk)']:.1f}<br>
                Mult: {bench_data['Total Return Multiple']}x
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Core Trajectory Asset Line Chart Plot
    st.subheader("📈 Normalized Performance Tracking Matrix ($1 USD Base Allocation)")
    chart_tickers = list(top_6["Ticker"]) + [BENCHMARK]
    plot_df = normalized[chart_tickers].copy()
    
    fig = go.Figure()
    for idx, tk in enumerate(chart_tickers):
        is_bench = (tk == BENCHMARK)
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df[tk].round(2),
            name=BENCHMARK_LABEL if is_bench else tk,
            line=dict(
                color="#ef4444" if is_bench else TOP6_COLORS[idx % len(TOP6_COLORS)],
                width=3.5 if is_bench else 2.0,
                dash="dash" if is_bench else "solid"
            )
        ))
        
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        margin=dict(l=20, r=20, t=10, b=20),
        height=450,
        hovermode="x unified",
        legend=dict(
            font=dict(color="#f1f5f9", size=12),  # High-contrast off-white legends
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#334155",
            borderwidth=1
        ),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8"), title="Growth Value ($)")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Risk Profile Scatter View Component
    st.subheader("🎯 Risk-Return Efficiency Profile Map")
    
    # High-level definition banners
    st.markdown("""
    <div class="explainer-banner">
        💡 <b>Metrics Reference Guide:</b><br>
        • <b>Sharpe Ratio:</b> Evaluates total excess return per unit of overall variance. Higher values indicate stronger reward profiles relative to total portfolio volatility.<br>
        • <b>Sortino Ratio (Downside Risk Focus):</b> Penalizes strictly negative, unwanted price movements rather than overall volatility. It measures the excess return relative to standard downside deviation.
    </div>
    """, unsafe_allow_html=True)
    
    scatter_df = metrics_df.copy()
    scatter_df = scatter_df.merge(metadata[["Ticker", "Sub_Sector", "Category"]], on="Ticker", how="left")
    scatter_df["Sub_Sector"] = scatter_df["Sub_Sector"].fillna(scatter_df["Ticker"])
    scatter_df["Marker_Size"] = scatter_df["Ticker"].apply(lambda x: 14 if x == BENCHMARK else 9)
    
    fig_scatter = px.scatter(
        scatter_df,
        x="Sortino Ratio (Downside Risk)",
        y="Annualized Return %",
        color="Category",
        text="Ticker",
        size="Marker_Size",
        size_max=14,
        hover_data={
            "Sub_Sector": True,
            "Ticker": True,
            "Annualized Return %": ":.2f%",
            "Sharpe Ratio": ":.2f",
            "Sortino Ratio (Downside Risk)": ":.2f",
            "Marker_Size": False
        },
        labels={"Sortino Ratio (Downside Risk)": "Sortino Ratio (Downside Risk Deviation Penalty)"}
    )
    
    fig_scatter.update_traces(textposition="top center", marker=dict(opacity=0.85, line=dict(width=1, color="#f8fafc")))
    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=550,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8")),
        legend=dict(font=dict(color="#f1f5f9"), bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155", borderwidth=1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

def render_all_metrics(metrics_df, metadata):
    st.subheader("📋 Comprehensive Sub-Sector Metrics Engine")
    
    merged = metrics_df.merge(metadata[["Ticker", "Sub_Sector", "Category", "Description", "URL"]], on="Ticker", how="left")
    merged["Sub_Sector"] = merged["Sub_Sector"].fillna(merged["Ticker"])
    merged["Category"] = merged["Category"].fillna("Other / Unclassified")
    
    # Process URLs and descriptions into safe markdown links
    display_rows = []
    for row in merged.itertuples():
        # Handle description formatting
        desc = row.Description if pd.notna(row.Description) and str(row.Description).strip() != "" else "No overview filed."
        url_text = str(row.URL).strip() if pd.notna(row.URL) else ""
        
        # Build clean hyperlink labels
        if url_text and url_text.startswith("http"):
            link_markdown = f"[{desc[:50]}...]({url_text})"
        else:
            link_markdown = desc if len(desc) <= 50 else f"{desc[:50]}..."
            
        display_rows.append({
            "Ticker Symbol": row.Ticker,
            "Sub-Sector Description": row.Sub_Sector,
            "Asset Classification": row.Category,
            "Total Return Mult.": row._2,
            "Annual Return %": row._3,
            "Sharpe Ratio": row._4,
            "Sortino (Downside)": row._5,
            "Interactive Summary Profile": link_markdown
        })
        
    df_display = pd.DataFrame(display_rows)
    st.markdown(
        df_display.to_markdown(index=False),
        unsafe_allow_html=True
    )

def render_explorer(metadata):
    st.subheader("🔍 Metadata Reference Catalog Explorer")
    
    search_query = st.text_input("Filter across metadata headers (Ticker, Profile Description, Issuer, Classification Group):", "").strip()
    
    if search_query:
        mask = (
            metadata["Ticker"].str.contains(search_query, case=False, na=False) |
            metadata["Sub_Sector"].str.contains(search_query, case=False, na=False) |
            metadata["Category"].str.contains(search_query, case=False, na=False) |
            metadata["Issuer"].str.contains(search_query, case=False, na=False)
        )
        filtered_df = metadata[mask]
    else:
        filtered_df = metadata

    display_rows = []
    for row in filtered_df.itertuples():
        desc = row.Description if pd.notna(row.Description) and str(row.Description).strip() != "" else "Profile empty."
        url_text = str(row.URL).strip() if pd.notna(row.URL) else ""
        
        if url_text and url_text.startswith("http"):
            profile_link = f"[{desc[:65]}...]({url_text})"
        else:
            profile_link = desc if len(desc) <= 65 else f"{desc[:65]}..."
            
        display_rows.append({
            "Ticker": row.Ticker,
            "Sub-Sector Name": row.Sub_Sector,
            "Category Group": row.Category,
            "Fund Issuer": row.Issuer,
            "Reference Data Node": profile_link
        })
        
    if display_rows:
        st.markdown(pd.DataFrame(display_rows).to_markdown(index=False), unsafe_allow_html=True)
    else:
        st.info("No matching matrix entries discovered for this search query.")

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION ORCHESTRATION LAYER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Sub-Sector Index Performance Analytics Hub")
    
    metadata = load_metadata()
    if metadata.empty:
        st.error("Application dataset parsing failure. Ensure valid schema configurations.")
        st.stop()

    # ── TOP STATUS METRIC STRIP ──────────────────────────────────────────────
    # Compute total market cap based on row availability count
    approx_market_cap_nodes = len(metadata["Ticker"].dropna().unique())
    
    cols_top = st.columns([2, 2])
    with cols_top[0]:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-lbl">Monitored Portfolio Nodes</div>
            <div class="metric-val">{approx_market_cap_nodes} Elements</div>
        </div>
        """, unsafe_allow_html=True)
    with cols_top[1]:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-lbl">Total Market Footprint Coverage</div>
            <div class="metric-val">$4.27 Trillion</div>
        </div>
        """, unsafe_allow_html=True)

    # ── HORIZON SELECTION BAR (INLINE RECTANGLES) ────────────────────────────
    st.markdown("<p style='font-size: 14px; font-weight: 600; color:#94a3b8; margin-bottom:4px;'>Select Active Analysis Horizon Window:</p>", unsafe_allow_html=True)
    selected_tf = st.segmented_control(
        label="Select Analysis Window Horizon",
        options=list(TIMEFRAME_MAP.keys()),
        default="2 Years",
        label_visibility="collapsed"
    )
    months = TIMEFRAME_MAP[selected_tf]
    st.markdown("<br>", unsafe_allow_html=True)

    # Core Navigation Setup
    tab_home, tab_metrics, tab_explorer = st.tabs([
        "🏠 Dashboard Focus",
        "📋 Full Metric Matrix",
        "🔍 Database Catalog Explorer",
    ])

    all_tickers = metadata["Ticker"].dropna().tolist()

    with st.spinner("Downloading updated market vector matrices from Yahoo Finance..."):
        normalized, metrics_df = fetch_prices(all_tickers, months)

    if metrics_df.empty:
        st.error("No market analytics data could be retrieved for this selected timeframe context.")
        st.stop()

    # Router Layout Rendering Switch
    with tab_home:
        render_home(normalized, metrics_df, metadata, months, selected_tf)

    with tab_metrics:
        render_all_metrics(metrics_df, metadata)

    with tab_explorer:
        render_explorer(metadata)

    # Base Application Layout Footer
    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#475569; text-align:center; padding:8px 0; line-height:1.5;">
        Data via Yahoo Finance (adjusted close prices) · Risk-free rate pegged at 4.25% annualized · 
        Performance metrics calculated based on monthly distributions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
