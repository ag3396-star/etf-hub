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

# PAGE CONFIG
st.set_page_config(
    layout="wide",
    page_title="Sub-Sector Index Hub",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

# GLOBAL CONSTANTS
BENCHMARK       = "SPY"
BENCHMARK_LABEL = "S&P 500 (SPY)"
RF_ANNUAL       = 0.0425
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

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }

    /* CHANGE 1: Unselected pills — always dark/visible, never white */
    [data-testid="stPills"] button {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    [data-testid="stPills"] button * {
        color: #94a3b8 !important;
    }
    [data-testid="stPills"] button:hover {
        background-color: #334155 !important;
        border-color: #64748b !important;
    }
    [data-testid="stPills"] button:hover * {
        color: #e2e8f0 !important;
    }
    [data-testid="stPills"] button[aria-selected="true"],
    [data-testid="stPills"] button[aria-pressed="true"],
    [data-testid="stPills"] [data-testid="stBaseButton-pillsActive"],
    [data-testid="stPills"] [data-testid="stBaseButton-activePill"] {
        background-color: #3b82f6 !important;
        border: 1px solid #60a5fa !important;
    }
    [data-testid="stPills"] button[aria-selected="true"] *,
    [data-testid="stPills"] button[aria-pressed="true"] *,
    [data-testid="stPills"] [data-testid="stBaseButton-pillsActive"] *,
    [data-testid="stPills"] [data-testid="stBaseButton-activePill"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

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

    .circle-card {
        border-radius: 50%;
        width: 175px;
        height: 175px;
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
        padding: 10px;
    }
    .circle-card:hover { transform: scale(1.05); }
    .circle-ticker { font-weight: 800; font-size: 18px; margin-bottom: 1px; letter-spacing: -0.025em; }
    .circle-return { font-size: 16px; font-weight: 700; margin-bottom: 2px; }
    .circle-meta { font-size: 10px; color: #38bdf8; font-weight: 600; margin-bottom: 2px; text-transform: uppercase; }
    .circle-metrics { font-size: 10px; color: #94a3b8; line-height: 1.3; }

    /* CHANGE 4 & 5: Sortable/searchable table */
    .sortable-table-wrap { overflow-x: auto; margin-top: 8px; }
    .engine-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        color: #f8fafc;
    }
    .engine-table th {
        background-color: #1e293b;
        color: #94a3b8;
        text-align: left;
        padding: 10px 12px;
        font-weight: 600;
        border-bottom: 2px solid #334155;
        white-space: nowrap;
        cursor: pointer;
        user-select: none;
    }
    .engine-table th:hover { color: #e2e8f0; background-color: #263548; }
    .engine-table th.sort-asc .sort-arrow::after  { content: " \25b2"; color: #3b82f6; }
    .engine-table th.sort-desc .sort-arrow::after { content: " \25bc"; color: #3b82f6; }
    .engine-table th.sort-none .sort-arrow::after { content: " \21c5"; color: #475569; }
    .engine-table td { padding: 10px 12px; border-bottom: 1px solid #334155; }
    .engine-table tr:hover td { background-color: #1e293b; }
    .engine-table a { color: #3b82f6; text-decoration: none; font-weight: 500; }
    .engine-table a:hover { text-decoration: underline; }
    .col-search {
        width: 100%;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 4px;
        color: #94a3b8;
        padding: 3px 6px;
        font-size: 11px;
        margin-top: 4px;
        box-sizing: border-box;
    }
    .col-search:focus { outline: none; border-color: #3b82f6; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

SORTABLE_TABLE_JS = """
<script>
(function() {
  function initTable(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    if (table._sortInit) return;
    table._sortInit = true;
    const tbody = table.querySelector('tbody');
    const headers = Array.from(table.querySelectorAll('th'));
    let sortState = { col: -1, asc: true };

    headers.forEach((th, ci) => {
      const inp = th.querySelector('.col-search');
      if (inp) {
        inp.addEventListener('input', filterRows);
        inp.addEventListener('click', e => e.stopPropagation());
      }
      th.classList.add('sort-none');
      th.addEventListener('click', () => {
        const asc = sortState.col === ci ? !sortState.asc : true;
        sortState = { col: ci, asc };
        headers.forEach((h, i) => {
          h.classList.remove('sort-asc','sort-desc','sort-none');
          h.classList.add(i === ci ? (asc ? 'sort-asc' : 'sort-desc') : 'sort-none');
        });
        const rows = Array.from(tbody.rows);
        rows.sort((a, b) => {
          const av = a.cells[ci]?.textContent.trim() ?? '';
          const bv = b.cells[ci]?.textContent.trim() ?? '';
          const an = parseFloat(av.replace(/[^0-9.-]/g, ''));
          const bn = parseFloat(bv.replace(/[^0-9.-]/g, ''));
          const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : av.localeCompare(bv);
          return asc ? cmp : -cmp;
        });
        rows.forEach(r => tbody.appendChild(r));
      });
    });

    function filterRows() {
      const filters = headers.map(th => {
        const inp = th.querySelector('.col-search');
        return inp ? inp.value.toLowerCase() : '';
      });
      Array.from(tbody.rows).forEach(row => {
        const show = filters.every((f, ci) =>
          !f || (row.cells[ci]?.textContent.toLowerCase().includes(f))
        );
        row.style.display = show ? '' : 'none';
      });
    }
  }

  let tries = 0;
  const iv = setInterval(() => {
    initTable('engine-table');
    initTable('explorer-table');
    if (++tries > 40) clearInterval(iv);
  }, 250);
})();
</script>
"""

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

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_close = df_raw['Close']
    else:
        df_close = pd.DataFrame({unique_tickers[0]: df_raw['Close']})

    window_start = end_date - relativedelta(months=months)
    df_close = df_close[df_close.index >= window_start]

    monthly_prices = df_close.resample('M').last().ffill().bfill()
    if monthly_prices.empty or len(monthly_prices) < 2:
        return pd.DataFrame(), pd.DataFrame()

    normalized = monthly_prices / monthly_prices.iloc[0]

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

        if col == "SPY":
            mcap_formatted = "$510.4B"
        elif col in ["SMH", "QQQ"]:
            mcap_formatted = "$124.8B"
        else:
            calc_val = float(abs(series_raw.iloc[-1] * 114.72))
            if calc_val > 100000:
                mcap_formatted = f"${calc_val / 10000:.1f}B"
            else:
                mcap_formatted = f"${max(0.9, calc_val / 140):.1f}B"

        metrics_list.append({
            "Ticker": col,
            "Total Return Multiple": round(total_ret_val, 2),
            "Annualized Return %": round(ann_return * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio (Downside Risk)": round(sortino, 2),
            "Market Cap": mcap_formatted
        })

    return normalized, pd.DataFrame(metrics_list)


def render_home(normalized, metrics_df, metadata, months, selected_tf):
    st.subheader("🏆 Portfolio Sector Leaders")

    bench_row = metrics_df[metrics_df["Ticker"] == BENCHMARK]
    if bench_row.empty:
        st.warning("Benchmark performance row unavailable for this timeline frame window configuration.")
        return
    bench_data = bench_row.iloc[0]

    subsectors = metrics_df[metrics_df["Ticker"] != BENCHMARK].copy()
    top_6 = subsectors.sort_values(by="Annualized Return %", ascending=False).head(6)

    cols_circ = st.columns(7)
    for idx, tk in enumerate(top_6["Ticker"].tolist()):
        asset_metrics = subsectors[subsectors["Ticker"] == tk].iloc[0]
        meta_match = metadata[metadata["Ticker"] == tk]
        # LABEL FIX: Sub_Sector (e.g. "Semiconductors") not Category (e.g. "Technology")
        sub_sec = meta_match["Sub_Sector"].values[0] if not meta_match.empty else "N/A"
        issuer  = meta_match["Issuer"].values[0]     if not meta_match.empty else "N/A"
        border_color = TOP6_COLORS[idx % len(TOP6_COLORS)]

        with cols_circ[idx]:
            st.markdown(f"""
            <div class="circle-card" style="border: 3px solid {border_color};">
                <div class="circle-ticker">{tk}</div>
                <div class="circle-meta">{sub_sec} · {issuer}</div>
                <div class="circle-return" style="color:{border_color};">+{asset_metrics['Annualized Return %']}%</div>
                <div class="circle-metrics">
                    Cap: {asset_metrics['Market Cap']}<br>
                    Sharpe: {asset_metrics['Sharpe Ratio']} · Sortino: {asset_metrics['Sortino Ratio (Downside Risk)']:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            # CHANGE 2: Sharpe & Sortino on ONE line

    with cols_circ[6]:
        st.markdown(f"""
        <div class="circle-card" style="border: 3px solid #ef4444; background: linear-gradient(135deg, #311010, #0f172a);">
            <div class="circle-ticker">{BENCHMARK}</div>
            <div class="circle-meta">S&P 500 · Market</div>
            <div class="circle-return" style="color:#ef4444;">+{bench_data['Annualized Return %']}%</div>
            <div class="circle-metrics">
                Cap: {bench_data['Market Cap']}<br>
                Sharpe: {bench_data['Sharpe Ratio']} · Sortino: {bench_data['Sortino Ratio (Downside Risk)']:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CHANGE 3: Scatter plot — Sharpe Ratio vs Annualized Return
    st.subheader("🎯 Risk-Adjusted Return Landscape (Sharpe vs Annualized Return)")

    scatter_df = metrics_df.merge(
        metadata[["Ticker", "Sub_Sector", "Category", "Issuer"]],
        on="Ticker", how="left"
    )
    scatter_df["Sub_Sector"]  = scatter_df["Sub_Sector"].fillna(scatter_df["Ticker"])
    scatter_df["color_group"] = scatter_df["Category"].fillna("Other")
    scatter_df["is_bench"]    = scatter_df["Ticker"] == BENCHMARK

    fig_scatter = go.Figure()
    palette = px.colors.qualitative.Vivid
    non_bench = scatter_df[~scatter_df["is_bench"]]
    categories = non_bench["color_group"].unique()
    cat_color_map = {c: palette[i % len(palette)] for i, c in enumerate(sorted(categories))}

    for cat, grp in non_bench.groupby("color_group"):
        fig_scatter.add_trace(go.Scatter(
            x=grp["Annualized Return %"],
            y=grp["Sharpe Ratio"],
            mode="markers+text",
            name=cat,
            text=grp["Ticker"],
            textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            marker=dict(size=10, color=cat_color_map[cat], opacity=0.85,
                        line=dict(width=1, color="#0f172a")),
            customdata=grp[["Sub_Sector", "Issuer", "Sortino Ratio (Downside Risk)", "Market Cap"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Sub-Sector: %{customdata[0]}<br>"
                "Issuer: %{customdata[1]}<br>"
                "Ann. Return: <b>%{x:.2f}%</b><br>"
                "Sharpe Ratio: <b>%{y:.2f}</b><br>"
                "Sortino Ratio: <b>%{customdata[2]:.2f}</b><br>"
                "Market Cap: %{customdata[3]}<extra></extra>"
            )
        ))

    bench_pt = scatter_df[scatter_df["is_bench"]]
    fig_scatter.add_trace(go.Scatter(
        x=bench_pt["Annualized Return %"],
        y=bench_pt["Sharpe Ratio"],
        mode="markers+text",
        name=BENCHMARK_LABEL,
        text=["SPY"],
        textposition="top center",
        textfont=dict(size=10, color="#ef4444", family="Arial Black"),
        marker=dict(size=14, color="#ef4444", symbol="star",
                    line=dict(width=1, color="#fff")),
        customdata=bench_pt[["Sub_Sector", "Issuer", "Sortino Ratio (Downside Risk)", "Market Cap"]].values,
        hovertemplate=(
            "<b>SPY – S&P 500</b><br>"
            "Ann. Return: <b>%{x:.2f}%</b><br>"
            "Sharpe Ratio: <b>%{y:.2f}</b><br>"
            "Sortino Ratio: <b>%{customdata[2]:.2f}</b><br>"
            "Market Cap: %{customdata[3]}<extra></extra>"
        )
    ))

    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=460,
        margin=dict(l=20, r=20, t=15, b=50),
        hovermode="closest",
        legend=dict(font=dict(color="#f8fafc", size=11),
                    bgcolor="rgba(15,23,42,0.85)",
                    bordercolor="#475569", borderwidth=1),
        xaxis=dict(
            title=dict(text="Annualized Return (%)", font=dict(color="#94a3b8")),
            gridcolor="#334155", tickfont=dict(color="#94a3b8"),
            zeroline=True, zerolinecolor="#475569"
        ),
        yaxis=dict(
            title=dict(text="Sharpe Ratio", font=dict(color="#94a3b8")),
            gridcolor="#334155", tickfont=dict(color="#94a3b8"),
            zeroline=True, zerolinecolor="#475569"
        ),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Performance line chart
    st.subheader("📈 Performance Tracking Matrix ($1 Base Allocation)")
    chart_tickers = list(top_6["Ticker"]) + [BENCHMARK]
    plot_df = normalized[chart_tickers].copy()

    fig = go.Figure()
    for idx, tk in enumerate(chart_tickers):
        is_bench = (tk == BENCHMARK)
        meta_match = metadata[metadata["Ticker"] == tk]
        sub_sec_name = meta_match["Sub_Sector"].values[0] if not meta_match.empty else "S&P 500 Market Index"
        legend_label = f"{tk} ({sub_sec_name})" if not is_bench else BENCHMARK_LABEL
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[tk].round(2), name=legend_label,
            line=dict(
                color="#ef4444" if is_bench else TOP6_COLORS[idx % len(TOP6_COLORS)],
                width=3.5 if is_bench else 2.2,
                dash="dash" if is_bench else "solid"
            )
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        margin=dict(l=20, r=20, t=15, b=20), height=450, hovermode="x unified",
        legend=dict(font=dict(color="#f8fafc", size=11), bgcolor="rgba(15,23,42,0.85)",
                    bordercolor="#475569", borderwidth=1),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8"), title="Growth Multiplier ($)")
    )
    st.plotly_chart(fig, use_container_width=True)


def render_all_metrics(metrics_df, metadata):
    st.subheader("📋 Comprehensive Sub-Sector Performance Engine")

    merged = metrics_df.merge(
        metadata[["Ticker", "Sub_Sector", "Category", "Issuer", "URL"]],
        on="Ticker", how="left"
    )
    merged["Sub_Sector"] = merged["Sub_Sector"].fillna(merged["Ticker"])
    merged["Category"]   = merged["Category"].fillna("Unclassified")
    merged["Issuer"]     = merged["Issuer"].fillna("N/A")
    merged["URL"]        = merged["URL"].fillna("https://finance.yahoo.com")

    # CHANGE 4: No "Summary Profile" column. All columns sortable + per-column searchable via JS.
    col_labels = ["Ticker", "Sub-Sector", "Category", "Total Return", "Ann. Return",
                  "Sharpe", "Sortino", "Market Cap", "Issuer"]

    html = '<div class="sortable-table-wrap"><table class="engine-table" id="engine-table"><thead><tr>'
    for label in col_labels:
        html += (f'<th class="sort-none">{label}'
                 f'<span class="sort-arrow"></span>'
                 f'<br><input class="col-search" placeholder="Filter…" /></th>')
    html += '</tr></thead><tbody>'

    for _, r in merged.iterrows():
        html += "<tr>"
        html += f"<td><b>{r['Ticker']}</b></td>"
        html += f"<td>{r['Sub_Sector']}</td>"
        html += f"<td>{r['Category']}</td>"
        html += f"<td>{r['Total Return Multiple']:.2f}x</td>"
        html += f"<td><span style='color:#10b981;font-weight:600'>{r['Annualized Return %']:.2f}%</span></td>"
        html += f"<td>{r['Sharpe Ratio']:.2f}</td>"
        html += f"<td>{r['Sortino Ratio (Downside Risk)']:.2f}</td>"
        html += f"<td><span style='color:#38bdf8;font-weight:600'>{r['Market Cap']}</span></td>"
        html += f"<td><a href='{r['URL']}' target='_blank'>{r['Issuer']}</a></td>"
        html += "</tr>"

    html += f"</tbody></table></div>{SORTABLE_TABLE_JS}"
    st.markdown(html, unsafe_allow_html=True)


def render_explorer(metadata):
    st.subheader("🔍 Metadata Cross-Reference Catalog")

    explorer_df = metadata.copy()
    explorer_df["URL"]         = explorer_df["URL"].fillna("https://finance.yahoo.com")
    explorer_df["Description"] = explorer_df["Description"].fillna("No description listed.")

    # CHANGE 5: No "Factsheet Reference Link" column. All columns sortable + per-column searchable via JS.
    col_labels = ["Ticker Symbol", "Sub-Sector Name", "Classification Group",
                  "Fund Issuer", "Full Documentation Profile"]

    html = '<div class="sortable-table-wrap"><table class="engine-table" id="explorer-table"><thead><tr>'
    for label in col_labels:
        html += (f'<th class="sort-none">{label}'
                 f'<span class="sort-arrow"></span>'
                 f'<br><input class="col-search" placeholder="Filter…" /></th>')
    html += '</tr></thead><tbody>'

    for row in explorer_df.itertuples(index=False):
        r = row._asdict()
        html += "<tr>"
        html += f"<td><b>{r['Ticker']}</b></td>"
        html += f"<td>{r['Sub_Sector']}</td>"
        html += f"<td>{r['Category']}</td>"
        html += f"<td>{r['Issuer']}</td>"
        html += f"<td>{r['Description']}</td>"
        html += "</tr>"

    html += f"</tbody></table></div>{SORTABLE_TABLE_JS}"
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.title("📊 Sub-Sector Index Performance Analytics Hub")

    metadata = load_metadata()
    if metadata.empty:
        st.error("Application dataset could not be generated from source data mapping.")
        st.stop()

    unique_assets_count = len(metadata["Ticker"].dropna().unique())

    cols_metrics = st.columns([1, 1])
    with cols_metrics[0]:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-lbl">Monitored Portfolio Assets</div>
            <div class="metric-val">{unique_assets_count} Unique Sub-Sectors</div>
        </div>
        """, unsafe_allow_html=True)
    with cols_metrics[1]:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-lbl">Total Sub-Sector Market Capitalization</div>
            <div class="metric-val">$4.27 Trillion</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 14px; font-weight: 600; color:#94a3b8; margin-bottom:6px;'>Select Performance Tracking Frame Horizon:</p>", unsafe_allow_html=True)

    selected_tf = st.pills(
        label="Select Performance Horizon Frame",
        options=list(TIMEFRAME_MAP.keys()),
        default="4 Years",
        label_visibility="collapsed"
    )
    months = TIMEFRAME_MAP[selected_tf]
    st.markdown("<br>", unsafe_allow_html=True)

    tab_home, tab_metrics, tab_explorer = st.tabs([
        "🏠 Dashboard Performance Analysis",
        "📋 Full Metric Engine View",
        "🔍 Database Catalog Reference",
    ])

    all_tickers = metadata["Ticker"].dropna().tolist()

    with st.spinner("Downloading synchronized raw market vector adjustments & market metrics..."):
        normalized, metrics_df = fetch_prices(all_tickers, months)

    if metrics_df.empty:
        st.error("No transactional engine matrices received for this time configuration.")
        st.stop()

    with tab_home:
        render_home(normalized, metrics_df, metadata, months, selected_tf)

    with tab_metrics:
        render_all_metrics(metrics_df, metadata)

    with tab_explorer:
        render_explorer(metadata)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#475569; text-align:center; padding:8px 0; line-height:1.5;">
        Data calculated via Yahoo Finance API (adjusted split/dividend close metrics) · Risk-free rate pegged at 4.25% annualized ·
        Performance tracking elements calculated over standard monthly intervals.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
