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

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Sub-Sector Index Hub",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CONSTANTS ──────────────────────────────────────────────────────────
BENCHMARK       = "VOO"           # CHANGE 1: VOO instead of SPY
BENCHMARK_LABEL = "S&P 500 (VOO)"
RF_ANNUAL       = 0.0425
CSV_PATH        = "us_subsector_etfs.csv"

TIMEFRAME_MAP = {
    "6 Years":  72,
    "5 Years":  60,
    "4 Years":  48,
    "3 Years":  36,
    "2 Years":  24,
    "1 Year":   12,
    "6 Months":  6,
    "3 Months":  3,
}

TOP6_COLORS = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899", "#06b6d4"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp { background-color: #0f172a; color: #f8fafc; }

    /* ── CHANGE 2: mobile-friendly viewport ── */
    @media (max-width: 430px) {
        .stApp { font-size: 13px; }
        .circle-card { width: 110px !important; height: 110px !important; padding: 6px !important; }
        .circle-ticker { font-size: 13px !important; }
        .circle-return { font-size: 12px !important; }
        .circle-meta   { font-size: 8px  !important; }
        .circle-metrics{ font-size: 8px  !important; }
        .metric-val    { font-size: 18px !important; }
        .sortable-table-wrap { font-size: 11px; }
    }

    /* ── Pills: force dark background on unselected, nuke Streamlit inline styles ── */
    [data-testid="stPills"] button,
    [data-testid="stPills"] button:not([aria-selected="true"]):not([aria-pressed="true"]),
    button[data-testid="stBaseButton-pills"],
    button[data-testid="stBaseButton-secondaryPills"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 999px !important;
        color: #94a3b8 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stPills"] button span,
    [data-testid="stPills"] button p,
    [data-testid="stPills"] button div,
    button[data-testid="stBaseButton-pills"] span,
    button[data-testid="stBaseButton-pills"] p {
        color: #94a3b8 !important;
        visibility: visible !important;
    }
    [data-testid="stPills"] button:hover {
        background-color: #334155 !important;
        background: #334155 !important;
        border-color: #64748b !important;
    }
    [data-testid="stPills"] button:hover span,
    [data-testid="stPills"] button:hover p { color: #e2e8f0 !important; }
    [data-testid="stPills"] button[aria-selected="true"],
    [data-testid="stPills"] button[aria-pressed="true"],
    button[data-testid="stBaseButton-pillsActive"],
    button[data-testid="stBaseButton-activePill"] {
        background-color: #3b82f6 !important;
        background: #3b82f6 !important;
        border: 1px solid #60a5fa !important;
    }
    [data-testid="stPills"] button[aria-selected="true"] span,
    [data-testid="stPills"] button[aria-selected="true"] p,
    [data-testid="stPills"] button[aria-pressed="true"] span,
    [data-testid="stPills"] button[aria-pressed="true"] p,
    button[data-testid="stBaseButton-pillsActive"] span,
    button[data-testid="stBaseButton-pillsActive"] p,
    button[data-testid="stBaseButton-activePill"] span,
    button[data-testid="stBaseButton-activePill"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ── Metric Cards ── */
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

    /* ── CHANGE 3: Glossary bar ── */
    .glossary-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0 18px 0;
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.5;
    }
    .glossary-item { display: flex; gap: 5px; align-items: baseline; }
    .glossary-term { color: #38bdf8; font-weight: 700; white-space: nowrap; }

    /* ── Circle Cards ── */
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
    .circle-ticker  { font-weight: 800; font-size: 18px; margin-bottom: 1px; letter-spacing: -0.025em; }
    .circle-return  { font-size: 16px; font-weight: 700; margin-bottom: 2px; }
    .circle-meta    { font-size: 10px; color: #38bdf8; font-weight: 600; margin-bottom: 2px; text-transform: uppercase; }
    .circle-metrics { font-size: 10px; color: #94a3b8; line-height: 1.3; }

    /* ── CHANGES 5 & 6: Sortable / searchable tables ── */
    .sortable-table-wrap { overflow-x: auto; margin-top: 4px; }
    .engine-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        color: #f8fafc;
    }
    .engine-table th {
        background-color: #1e293b;
        color: #cbd5e1;
        text-align: left;
        padding: 10px 12px 6px 12px;
        font-weight: 700;
        border-bottom: 2px solid #334155;
        white-space: nowrap;
        cursor: pointer;
        user-select: none;
        vertical-align: top;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .engine-table th:hover { color: #f8fafc; background-color: #263548; }
    /* Sort arrow always visible in the header row */
    .th-label { display: flex; align-items: center; gap: 4px; margin-bottom: 4px; }
    .sort-icon {
        display: inline-block;
        font-size: 12px;
        color: #475569;
        transition: color 0.15s;
        min-width: 14px;
    }
    .engine-table th.sort-asc  .sort-icon { color: #3b82f6; }
    .engine-table th.sort-desc .sort-icon { color: #3b82f6; }
    .engine-table td { padding: 10px 12px; border-bottom: 1px solid #334155; vertical-align: middle; }
    .engine-table tr:hover td { background-color: #1a2540; }
    .engine-table a { color: #3b82f6; text-decoration: none; font-weight: 500; }
    .engine-table a:hover { text-decoration: underline; }
    /* Per-column filter input */
    .col-search {
        width: 100%;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 4px;
        color: #94a3b8;
        padding: 4px 7px;
        font-size: 11px;
        box-sizing: border-box;
    }
    .col-search:focus { outline: none; border-color: #3b82f6; color: #f8fafc; }
    .col-search::placeholder { color: #475569; }

    /* ── CHANGE 4: Footer attribution ── */
    .footer-bar {
        margin-top: 24px;
        padding: 16px 0 8px 0;
        border-top: 1px solid #1e293b;
        text-align: center;
        font-size: 11px;
        color: #475569;
        line-height: 1.8;
    }
    .footer-bar a { color: #38bdf8; text-decoration: none; }
    .footer-bar a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ── JAVASCRIPT: sortable + per-column live search ─────────────────────────────
# CHANGES 5 & 6: Rewrote the JS to be more robust.
# Key fix: we use a data-col-index attribute on each input to avoid relying on
# DOM hierarchy traversal that breaks inside Streamlit's sandboxed iframe.
# The MutationObserver re-inits when new tables appear (tab switch).
SORTABLE_TABLE_JS = """
<script>
(function() {
  'use strict';

  function initTable(tableId) {
    const table = document.getElementById(tableId);
    if (!table || table._si) return;
    table._si = true;

    const tbody  = table.tBodies[0];
    const ths    = Array.from(table.tHead.rows[0].cells);
    let sortCol  = -1, sortAsc = true;

    // ── Wire up filter inputs ──────────────────────────────────────────────
    ths.forEach((th, ci) => {
      const inp = th.querySelector('.col-search');
      if (!inp) return;
      inp.setAttribute('data-ci', ci);
      inp.addEventListener('input', runFilter);
      inp.addEventListener('keydown', e => e.stopPropagation());
      inp.addEventListener('click',   e => e.stopPropagation());
    });

    // ── Wire up sort on header click ───────────────────────────────────────
    ths.forEach((th, ci) => {
      th.addEventListener('click', () => {
        sortAsc = (sortCol === ci) ? !sortAsc : true;
        sortCol = ci;
        // Update visual state
        ths.forEach((h, i) => {
          h.classList.remove('sort-asc', 'sort-desc', 'sort-none');
          h.classList.add(i === ci ? (sortAsc ? 'sort-asc' : 'sort-desc') : 'sort-none');
          const icon = h.querySelector('.sort-icon');
          if (!icon) return;
          if (i === ci) { icon.textContent = sortAsc ? '▲' : '▼'; }
          else          { icon.textContent = '⇅'; }
        });
        // Sort rows
        const rows = Array.from(tbody.rows).filter(r => r.style.display !== 'none' || true);
        rows.sort((a, b) => {
          const at = a.cells[ci] ? a.cells[ci].textContent.trim() : '';
          const bt = b.cells[ci] ? b.cells[ci].textContent.trim() : '';
          const an = parseFloat(at.replace(/[^0-9.\\-]/g, ''));
          const bn = parseFloat(bt.replace(/[^0-9.\\-]/g, ''));
          const cmp = (!isNaN(an) && !isNaN(bn)) ? (an - bn) : at.localeCompare(bt);
          return sortAsc ? cmp : -cmp;
        });
        rows.forEach(r => tbody.appendChild(r));
        runFilter(); // re-apply filters after sort
      });
    });

    // ── Filter function ────────────────────────────────────────────────────
    function runFilter() {
      const filters = ths.map(th => {
        const inp = th.querySelector('.col-search');
        return inp ? inp.value.trim().toLowerCase() : '';
      });
      const hasFilter = filters.some(f => f.length > 0);
      Array.from(tbody.rows).forEach(row => {
        if (!hasFilter) { row.style.display = ''; return; }
        const match = filters.every((f, ci) => {
          if (!f) return true;
          const cell = row.cells[ci];
          return cell && cell.textContent.toLowerCase().includes(f);
        });
        row.style.display = match ? '' : 'none';
      });
    }
  }

  // Poll + observe to handle Streamlit's async tab rendering
  function tryInit() {
    initTable('engine-table');
    initTable('explorer-table');
  }
  tryInit();
  const obs = new MutationObserver(tryInit);
  obs.observe(document.body, { childList: true, subtree: true });
  // Safety net polling
  let n = 0;
  const iv = setInterval(() => { tryInit(); if (++n > 60) clearInterval(iv); }, 300);
})();
</script>
"""

# ── PILL JS FIXER ─────────────────────────────────────────────────────────────
def inject_pill_style_js():
    st.markdown("""
    <script>
    (function fixPills() {
        function applyStyles() {
            const container = document.querySelector('[data-testid="stPills"]');
            if (!container) return;
            container.querySelectorAll('button').forEach(btn => {
                const active = btn.getAttribute('aria-selected') === 'true'
                            || btn.getAttribute('aria-pressed')  === 'true'
                            || (btn.dataset.testid || '').includes('Active')
                            || (btn.dataset.testid || '').includes('active');
                if (active) {
                    btn.style.setProperty('background',       '#3b82f6',         'important');
                    btn.style.setProperty('background-color', '#3b82f6',         'important');
                    btn.style.setProperty('border',           '1px solid #60a5fa','important');
                    btn.querySelectorAll('*').forEach(el => el.style.setProperty('color', '#ffffff', 'important'));
                } else {
                    btn.style.setProperty('background',       '#1e293b',         'important');
                    btn.style.setProperty('background-color', '#1e293b',         'important');
                    btn.style.setProperty('border',           '1px solid #475569','important');
                    btn.querySelectorAll('*').forEach(el => el.style.setProperty('color', '#94a3b8', 'important'));
                }
            });
        }
        applyStyles();
        new MutationObserver(applyStyles).observe(document.body, { childList:true, subtree:true, attributes:true });
        let t=0; const iv=setInterval(()=>{ applyStyles(); if(++t>30) clearInterval(iv); }, 200);
    })();
    </script>
    """, unsafe_allow_html=True)

# ── DATA LOADERS ──────────────────────────────────────────────────────────────
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
        return pd.DataFrame(columns=["#","Ticker","Sub_Sector","Category","Issuer","Description","URL"])

@st.cache_data(show_spinner=False)
def _fetch_aum(tickers):
    """Fetch live AUM (totalAssets) for a list of ETF tickers via yfinance.
    Returns a dict {ticker: formatted_string}.  Falls back gracefully on errors.
    """
    aum_map = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).fast_info
            # fast_info exposes .market_cap for stocks; ETFs expose totalAssets via .info
            raw = getattr(info, 'total_assets', None)
            if raw is None:
                # fallback to slower .info dict
                raw = yf.Ticker(tk).info.get('totalAssets', None)
            if raw and raw > 0:
                if raw >= 1e9:
                    aum_map[tk] = f"${raw/1e9:.1f}B"
                elif raw >= 1e6:
                    aum_map[tk] = f"${raw/1e6:.0f}M"
                else:
                    aum_map[tk] = f"${raw:,.0f}"
            else:
                aum_map[tk] = "N/A"
        except Exception:
            aum_map[tk] = "N/A"
    return aum_map


@st.cache_data(show_spinner=False)
def fetch_prices(tickers, months):
    end_date   = datetime.today()
    start_date = end_date - relativedelta(months=months + 1)
    unique_tickers = list(set(tickers + [BENCHMARK]))

    try:
        df_raw = yf.download(unique_tickers, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)
    except Exception as e:
        st.error(f"Yahoo Finance download failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_close = df_raw['Close'] if isinstance(df_raw.columns, pd.MultiIndex) \
               else pd.DataFrame({unique_tickers[0]: df_raw['Close']})

    window_start  = end_date - relativedelta(months=months)
    df_close      = df_close[df_close.index >= window_start]
    monthly_prices = df_close.resample('M').last().ffill().bfill()

    if monthly_prices.empty or len(monthly_prices) < 2:
        return pd.DataFrame(), pd.DataFrame()

    normalized  = monthly_prices / monthly_prices.iloc[0]
    rf_monthly  = RF_ANNUAL / 12
    n_years     = months / 12.0
    metrics_list = []

    # Fetch live AUM for all tickers in one cached batch
    aum_map = _fetch_aum(tuple(normalized.columns.tolist()))

    for col in normalized.columns:
        series_norm = normalized[col]
        series_raw  = monthly_prices[col]
        m_returns   = series_raw.pct_change().dropna()
        if m_returns.empty:
            continue

        total_ret = series_norm.iloc[-1] / series_norm.iloc[0]
        ann_ret   = (total_ret ** (1 / n_years)) - 1 if total_ret > 0 else 0
        exc_ret   = m_returns - rf_monthly
        vol       = m_returns.std() * np.sqrt(12)
        sharpe    = (exc_ret.mean() * 12) / vol if vol > 0.0001 else 0

        dn = m_returns[m_returns < 0]
        if len(dn) < 2:
            sortino = 99.9
        else:
            dv = dn.std() * np.sqrt(12)
            sortino = (exc_ret.mean() * 12) / dv if dv > 0.0001 else 0

        # Dynamic AUM from yfinance — no hardcoding
        mcap_fmt = aum_map.get(col, "N/A")

        metrics_list.append({
            "Ticker":                       col,
            "Total Return Multiple":        round(total_ret, 2),
            "Annualized Return %":          round(ann_ret * 100, 2),
            "Sharpe Ratio":                 round(sharpe, 2),
            "Sortino Ratio (Downside Risk)":round(sortino, 2),
            "Market Cap":                   mcap_fmt,
        })

    return normalized, pd.DataFrame(metrics_list)

# ── BUILD TABLE HTML helper ───────────────────────────────────────────────────
def _make_table(table_id, col_labels, rows_html):
    """Returns full HTML for a sortable+searchable table."""
    hdr = ""
    for label in col_labels:
        hdr += (
            f'<th class="sort-none">'
            f'<div class="th-label">{label}<span class="sort-icon">⇅</span></div>'
            f'<input class="col-search" placeholder="Filter…" />'
            f'</th>'
        )
    return (
        f'<div class="sortable-table-wrap">'
        f'<table class="engine-table" id="{table_id}">'
        f'<thead><tr>{hdr}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
        f'{SORTABLE_TABLE_JS}'
    )

# ── RENDER HOME ───────────────────────────────────────────────────────────────
def render_home(normalized, metrics_df, metadata, months, selected_tf):
    st.subheader("🏆 Portfolio Sector Leaders")

    bench_row = metrics_df[metrics_df["Ticker"] == BENCHMARK]
    if bench_row.empty:
        st.warning("Benchmark data unavailable for this timeframe.")
        return
    bench_data = bench_row.iloc[0]

    subsectors = metrics_df[metrics_df["Ticker"] != BENCHMARK].copy()
    top_6      = subsectors.sort_values("Annualized Return %", ascending=False).head(6)

    # ── CHANGE 3: Glossary bar ─────────────────────────────────────────────
    rf_pct = f"{RF_ANNUAL*100:.2f}%"
    st.markdown(f"""
    <div class="glossary-bar">
        <div class="glossary-item"><span class="glossary-term">Ann. Return</span><span>— compound yearly gain over the selected period, expressed as a % per year.</span></div>
        <div class="glossary-item"><span class="glossary-term">Sharpe</span><span>— excess return above the risk-free rate ({rf_pct} p.a.) divided by total volatility; higher = better reward per unit of risk.</span></div>
        <div class="glossary-item"><span class="glossary-term">Sortino</span><span>— like Sharpe but uses only downside volatility (negative months) in the denominator, using the same {rf_pct} risk-free rate; penalises bad volatility only.</span></div>
        <div class="glossary-item"><span class="glossary-term">AUM</span><span>— live Assets Under Management fetched from Yahoo Finance; reflects total investor capital in the ETF.</span></div>
        <div class="glossary-item" style="flex-basis:100%;margin-top:4px;border-top:1px solid #334155;padding-top:6px;">
            <span style="color:#f59e0b;font-weight:600;">Risk-Free Rate</span>
            <span>— all Sharpe &amp; Sortino ratios use <b style="color:#f8fafc;">{rf_pct} annualised</b> (approximating the US 3-month T-bill yield) as the baseline return an investor can earn with zero risk. Excess returns above this hurdle are what the ratios measure.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Circle cards ──────────────────────────────────────────────────────
    cols_circ = st.columns(7)
    for idx, tk in enumerate(top_6["Ticker"].tolist()):
        am = subsectors[subsectors["Ticker"] == tk].iloc[0]
        mm = metadata[metadata["Ticker"] == tk]
        sub_sec = mm["Sub_Sector"].values[0] if not mm.empty else "N/A"
        issuer  = mm["Issuer"].values[0]     if not mm.empty else "N/A"
        bc      = TOP6_COLORS[idx % len(TOP6_COLORS)]
        with cols_circ[idx]:
            st.markdown(f"""
            <div class="circle-card" style="border:3px solid {bc};">
                <div class="circle-ticker">{tk}</div>
                <div class="circle-meta">{sub_sec} · {issuer}</div>
                <div class="circle-return" style="color:{bc};">+{am['Annualized Return %']}%</div>
                <div class="circle-metrics">
                    Cap: {am['Market Cap']}<br>
                    Sharpe: {am['Sharpe Ratio']} · Sortino: {am['Sortino Ratio (Downside Risk)']:.1f}
                </div>
            </div>""", unsafe_allow_html=True)

    with cols_circ[6]:
        st.markdown(f"""
        <div class="circle-card" style="border:3px solid #ef4444;background:linear-gradient(135deg,#311010,#0f172a);">
            <div class="circle-ticker">{BENCHMARK}</div>
            <div class="circle-meta">S&P 500 · Vanguard</div>
            <div class="circle-return" style="color:#ef4444;">+{bench_data['Annualized Return %']}%</div>
            <div class="circle-metrics">
                Cap: {bench_data['Market Cap']}<br>
                Sharpe: {bench_data['Sharpe Ratio']} · Sortino: {bench_data['Sortino Ratio (Downside Risk)']:.1f}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter plot ──────────────────────────────────────────────────────
    st.subheader("🎯 Risk-Adjusted Return Landscape (Sharpe vs Annualized Return)")

    scatter_df = metrics_df.merge(
        metadata[["Ticker","Sub_Sector","Category","Issuer"]], on="Ticker", how="left"
    )
    scatter_df["Sub_Sector"]  = scatter_df["Sub_Sector"].fillna(scatter_df["Ticker"])
    scatter_df["color_group"] = scatter_df["Category"].fillna("Other")
    scatter_df["is_bench"]    = scatter_df["Ticker"] == BENCHMARK

    fig_s = go.Figure()
    palette = px.colors.qualitative.Vivid
    non_bench = scatter_df[~scatter_df["is_bench"]]
    cat_map = {c: palette[i % len(palette)] for i, c in enumerate(sorted(non_bench["color_group"].unique()))}

    for cat, grp in non_bench.groupby("color_group"):
        fig_s.add_trace(go.Scatter(
            x=grp["Annualized Return %"], y=grp["Sharpe Ratio"],
            mode="markers+text", name=cat,
            text=grp["Ticker"], textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            marker=dict(size=10, color=cat_map[cat], opacity=0.85, line=dict(width=1, color="#0f172a")),
            customdata=grp[["Sub_Sector","Issuer","Sortino Ratio (Downside Risk)","Market Cap"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>Sub-Sector: %{customdata[0]}<br>Issuer: %{customdata[1]}<br>"
                "Ann. Return: <b>%{x:.2f}%</b><br>Sharpe: <b>%{y:.2f}</b><br>"
                "Sortino: <b>%{customdata[2]:.2f}</b><br>AUM: %{customdata[3]}<extra></extra>"
            )
        ))

    bp = scatter_df[scatter_df["is_bench"]]
    fig_s.add_trace(go.Scatter(
        x=bp["Annualized Return %"], y=bp["Sharpe Ratio"],
        mode="markers+text", name=BENCHMARK_LABEL,
        text=[BENCHMARK], textposition="top center",
        textfont=dict(size=10, color="#ef4444", family="Arial Black"),
        marker=dict(size=14, color="#ef4444", symbol="star", line=dict(width=1, color="#fff")),
        customdata=bp[["Sub_Sector","Issuer","Sortino Ratio (Downside Risk)","Market Cap"]].values,
        hovertemplate=(
            f"<b>{BENCHMARK} – S&P 500 (Vanguard)</b><br>"
            "Ann. Return: <b>%{x:.2f}%</b><br>Sharpe: <b>%{y:.2f}</b><br>"
            "Sortino: <b>%{customdata[2]:.2f}</b><br>AUM: %{customdata[3]}<extra></extra>"
        )
    ))

    fig_s.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        height=460, margin=dict(l=20, r=20, t=15, b=50), hovermode="closest",
        legend=dict(font=dict(color="#f8fafc", size=11), bgcolor="rgba(15,23,42,0.85)",
                    bordercolor="#475569", borderwidth=1),
        xaxis=dict(title=dict(text="Annualized Return (%)", font=dict(color="#94a3b8")),
                   gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   zeroline=True, zerolinecolor="#475569"),
        yaxis=dict(title=dict(text="Sharpe Ratio", font=dict(color="#94a3b8")),
                   gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   zeroline=True, zerolinecolor="#475569"),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # ── Performance line chart ─────────────────────────────────────────────
    st.subheader("📈 Performance Tracking Matrix ($1 Base Allocation)")
    chart_tickers = list(top_6["Ticker"]) + [BENCHMARK]
    # Guard: only keep tickers that actually exist in normalized
    chart_tickers = [t for t in chart_tickers if t in normalized.columns]
    plot_df = normalized[chart_tickers].copy()

    fig = go.Figure()
    for idx, tk in enumerate(chart_tickers):
        is_bench = (tk == BENCHMARK)
        mm = metadata[metadata["Ticker"] == tk]
        sub_nm = mm["Sub_Sector"].values[0] if not mm.empty else "S&P 500 (Vanguard)"
        lbl    = BENCHMARK_LABEL if is_bench else f"{tk} ({sub_nm})"
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[tk].round(2), name=lbl,
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
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   title=dict(text="Growth Multiplier ($)", font=dict(color="#94a3b8")))
    )
    st.plotly_chart(fig, use_container_width=True)


# ── RENDER FULL METRICS TABLE ─────────────────────────────────────────────────
def render_all_metrics(metrics_df, metadata):
    st.subheader("📋 Comprehensive Sub-Sector Performance Engine")

    merged = metrics_df.merge(
        metadata[["Ticker","Sub_Sector","Category","Issuer","URL"]], on="Ticker", how="left"
    )
    merged["Sub_Sector"] = merged["Sub_Sector"].fillna(merged["Ticker"])
    merged["Category"]   = merged["Category"].fillna("Unclassified")
    merged["Issuer"]     = merged["Issuer"].fillna("N/A")
    merged["URL"]        = merged["URL"].fillna("https://finance.yahoo.com")

    col_labels = ["Ticker","Sub-Sector","Category","Total Return","Ann. Return",
                  "Sharpe","Sortino","Market Cap","Issuer"]
    rows = ""
    for _, r in merged.iterrows():
        rows += (
            f"<tr>"
            f"<td><b>{r['Ticker']}</b></td>"
            f"<td>{r['Sub_Sector']}</td>"
            f"<td>{r['Category']}</td>"
            f"<td>{r['Total Return Multiple']:.2f}x</td>"
            f"<td><span style='color:#10b981;font-weight:600'>{r['Annualized Return %']:.2f}%</span></td>"
            f"<td>{r['Sharpe Ratio']:.2f}</td>"
            f"<td>{r['Sortino Ratio (Downside Risk)']:.2f}</td>"
            f"<td><span style='color:#38bdf8;font-weight:600'>{r['Market Cap']}</span></td>"
            f"<td><a href='{r['URL']}' target='_blank'>{r['Issuer']}</a></td>"
            f"</tr>"
        )
    st.markdown(_make_table("engine-table", col_labels, rows), unsafe_allow_html=True)


# ── RENDER EXPLORER TABLE ─────────────────────────────────────────────────────
def render_explorer(metadata):
    st.subheader("🔍 Metadata Cross-Reference Catalog")

    df = metadata.copy()
    df["URL"]         = df["URL"].fillna("https://finance.yahoo.com")
    df["Description"] = df["Description"].fillna("No description listed.")

    col_labels = ["Ticker Symbol","Sub-Sector Name","Classification Group",
                  "Fund Issuer","Full Documentation Profile"]
    rows = ""
    for _, r in df.iterrows():
        rows += (
            f"<tr>"
            f"<td><b>{r['Ticker']}</b></td>"
            f"<td>{r['Sub_Sector']}</td>"
            f"<td>{r['Category']}</td>"
            f"<td>{r['Issuer']}</td>"
            f"<td>{r['Description']}</td>"
            f"</tr>"
        )
    st.markdown(_make_table("explorer-table", col_labels, rows), unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Sub-Sector Index Performance Analytics Hub")
    inject_pill_style_js()

    metadata = load_metadata()
    if metadata.empty:
        st.error("Application dataset could not be loaded.")
        st.stop()

    unique_assets_count = len(metadata["Ticker"].dropna().unique())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-lbl">Monitored Portfolio Assets</div>
            <div class="metric-val">{unique_assets_count} Unique Sub-Sectors</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-lbl">Total Sub-Sector Market Capitalization</div>
            <div class="metric-val">$4.27 Trillion</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:14px;font-weight:600;color:#94a3b8;margin-bottom:6px;'>"
                "Select Performance Tracking Frame Horizon:</p>", unsafe_allow_html=True)

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
    with st.spinner("Fetching market data…"):
        normalized, metrics_df = fetch_prices(all_tickers, months)

    if metrics_df.empty:
        st.error("No market data received for this timeframe.")
        st.stop()

    with tab_home:
        render_home(normalized, metrics_df, metadata, months, selected_tf)
    with tab_metrics:
        render_all_metrics(metrics_df, metadata)
    with tab_explorer:
        render_explorer(metadata)

    # ── CHANGE 4: Footer with attribution ─────────────────────────────────
    st.markdown("""
    <div class="footer-bar">
        Data sourced via Yahoo Finance API (adjusted close prices) · Performance computed over monthly intervals.<br>
        Risk-free rate: <b>4.25% p.a.</b> (US 3-month T-bill proxy) — used as the hurdle rate in all Sharpe &amp; Sortino calculations.<br>
        Built &amp; maintained by <a href="mailto:arjjun.garg@gmail.com">Arjun Garg</a>
        &nbsp;·&nbsp;
        <a href="mailto:arjjun.garg@gmail.com">arjjun.garg@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
