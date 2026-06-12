"""
Sub-Sector Index Performance Hub
=================================
A Streamlit application that analyzes US sub-sector ETF performance
across multiple time horizons using Yahoo Finance data.

Author: Arjun Garg
"""

import streamlit as st
import streamlit.components.v1 as components
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

# ── TABLE RENDERER ───────────────────────────────────────────────────────────
# Uses st.components.v1.html() so <script> tags actually execute.
# st.markdown strips scripts entirely — this is the only reliable approach.

def _render_table(table_id, col_labels, rows_html):
    """Render a fully interactive sortable+searchable table via components.html."""

    # Build header: each <th> has a clickable label+arrow div and a filter input
    hdr = ""
    for label in col_labels:
        hdr += (
            f'<th class="sort-none" onclick="sortBy(this)">'
            f'<div class="th-label">'
            f'  <span class="th-text">{label}</span>'
            f'  <span class="sort-icon">&#8693;</span>'
            f'</div>'
            f'<input class="col-search" placeholder="Filter…"'
            f'  oninput="filterTable()" onclick="event.stopPropagation()"'
            f'  onkeydown="event.stopPropagation()" />'
            f'</th>'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f172a;
    color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
  }}
  .wrap {{ overflow-x: auto; width: 100%; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    table-layout: auto;
  }}
  thead {{ position: sticky; top: 0; z-index: 10; }}
  th {{
    background: #1e293b;
    color: #cbd5e1;
    padding: 10px 12px 6px 12px;
    font-weight: 700;
    font-size: 13px;
    border-bottom: 2px solid #334155;
    white-space: nowrap;
    vertical-align: top;
    cursor: pointer;
    user-select: none;
  }}
  th:hover {{ background: #263548; color: #f8fafc; }}
  .th-label {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
  }}
  .th-text {{ flex: 1; }}
  /* Sort icon — always visible, large and clear */
  .sort-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    font-size: 15px;
    font-weight: 900;
    color: #475569;
    background: #0f172a;
    border-radius: 4px;
    border: 1px solid #334155;
    flex-shrink: 0;
    transition: all 0.15s;
  }}
  th.sort-asc  .sort-icon {{ color: #fff; background: #3b82f6; border-color: #3b82f6; }}
  th.sort-desc .sort-icon {{ color: #fff; background: #3b82f6; border-color: #3b82f6; }}
  /* Filter input */
  .col-search {{
    width: 100%;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #94a3b8;
    padding: 5px 8px;
    font-size: 12px;
  }}
  .col-search:focus {{ outline: none; border-color: #3b82f6; color: #f8fafc; }}
  .col-search::placeholder {{ color: #475569; }}
  td {{
    padding: 10px 12px;
    border-bottom: 1px solid #1e293b;
    vertical-align: middle;
  }}
  tr:hover td {{ background: #1a2540; }}
  a {{ color: #3b82f6; text-decoration: none; font-weight: 500; }}
  a:hover {{ text-decoration: underline; }}
  .pos {{ color: #10b981; font-weight: 600; }}
  .neg {{ color: #ef4444; font-weight: 600; }}
  .aum {{ color: #38bdf8; font-weight: 600; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<div class="wrap">
<table id="{table_id}">
  <thead><tr>{hdr}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>

<script>
var tbl      = document.getElementById('{table_id}');
var tbody    = tbl.tBodies[0];
var ths      = Array.prototype.slice.call(tbl.tHead.rows[0].cells);
var sortCol  = -1;
var sortAsc  = true;

/* ── Sort ── */
function sortBy(th) {{
  var ci = ths.indexOf(th);
  if (ci < 0) return;
  sortAsc = (sortCol === ci) ? !sortAsc : true;
  sortCol = ci;

  ths.forEach(function(h, i) {{
    h.classList.remove('sort-asc', 'sort-desc', 'sort-none');
    var icon = h.querySelector('.sort-icon');
    if (i === ci) {{
      h.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');
      /* ▲ = ascending (A→Z / low→high), ▼ = descending */
      icon.innerHTML = sortAsc ? '&#9650;' : '&#9660;';
    }} else {{
      h.classList.add('sort-none');
      icon.innerHTML = '&#8693;';
    }}
  }});

  var rows = Array.prototype.slice.call(tbody.rows);
  rows.sort(function(a, b) {{
    var at = a.cells[ci] ? a.cells[ci].textContent.trim() : '';
    var bt = b.cells[ci] ? b.cells[ci].textContent.trim() : '';
    /* strip non-numeric chars but keep minus & decimal */
    var an = parseFloat(at.replace(/[^0-9.\u002d]/g, ''));
    var bn = parseFloat(bt.replace(/[^0-9.\u002d]/g, ''));
    var cmp = (!isNaN(an) && !isNaN(bn)) ? (an - bn) : at.localeCompare(bt);
    return sortAsc ? cmp : -cmp;
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
  filterTable(); /* re-apply active filters after sort */
}}

/* ── Filter ── */
function filterTable() {{
  var filters = ths.map(function(th) {{
    var inp = th.querySelector('.col-search');
    return inp ? inp.value.trim().toLowerCase() : '';
  }});
  var anyActive = filters.some(function(f) {{ return f.length > 0; }});

  Array.prototype.slice.call(tbody.rows).forEach(function(row) {{
    if (!anyActive) {{ row.classList.remove('hidden'); return; }}
    var show = filters.every(function(f, ci) {{
      if (!f) return true;
      var cell = row.cells[ci];
      return cell && cell.textContent.toLowerCase().indexOf(f) >= 0;
    }});
    if (show) row.classList.remove('hidden');
    else      row.classList.add('hidden');
  }});
}}
</script>
</body>
</html>"""
    # Count rows to set a sensible iframe height (header ~90px + ~46px/row, max 800)
    n_rows = rows_html.count("<tr>")
    height = min(800, 90 + n_rows * 46)
    components.html(html, height=height, scrolling=True)


# ── PILL JS FIXER ─────────────────────────────────────────────────────────────
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
        height=520, margin=dict(l=20, r=20, t=15, b=20), hovermode="closest",
        legend=dict(
            orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
            font=dict(color="#f8fafc", size=11), bgcolor="rgba(15,23,42,0.85)",
            bordercolor="#475569", borderwidth=1,
        ),
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
        margin=dict(l=20, r=20, t=15, b=20), height=480, hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
            font=dict(color="#f8fafc", size=11), bgcolor="rgba(15,23,42,0.85)",
            bordercolor="#475569", borderwidth=1,
        ),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   title=dict(text="Growth Multiplier ($)", font=dict(color="#94a3b8")))
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── S&P 500 Top-6 Companies Section ───────────────────────────────────────
    render_sp500_section(months, selected_tf)


@st.cache_data(show_spinner=False)
def fetch_sp500_top6(months):
    """Fetch S&P 500 tickers, download price data, compute metrics, return top-6 + VOO."""
    import requests
    from io import StringIO as _StringIO
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers, timeout=15)
        tables = pd.read_html(_StringIO(resp.text))
        tickers = tables[0]['Symbol'].tolist()
        # Fix common ticker format issues
        tickers = [t.replace('.', '-') for t in tickers]
    except Exception:
        tickers = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","JPM","V"]

    end_date   = datetime.today()
    start_date = end_date - relativedelta(months=months + 1)
    dl_tickers = list(set(tickers + [BENCHMARK]))

    try:
        raw = yf.download(dl_tickers, start=start_date, end=end_date,
                          progress=False, auto_adjust=True, threads=True)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else pd.DataFrame({'close': raw['Close']})

    window_start = end_date - relativedelta(months=months)
    df_close = df_close[df_close.index >= window_start]
    monthly  = df_close.resample('ME').last().ffill().bfill()

    if monthly.empty or len(monthly) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    normalized_eq = monthly / monthly.iloc[0]
    rf_monthly    = RF_ANNUAL / 12
    n_years       = months / 12.0
    results = []

    for col in normalized_eq.columns:
        if col not in monthly.columns:
            continue
        s = monthly[col]
        mr = s.pct_change().dropna()
        if mr.empty or len(mr) < 2:
            continue
        total = normalized_eq[col].iloc[-1]
        ann   = (total ** (1 / n_years) - 1) * 100 if total > 0 else 0
        exc   = mr - rf_monthly
        vol   = mr.std() * np.sqrt(12)
        sharpe = (exc.mean() * 12) / vol if vol > 0.0001 else 0
        dn = mr[mr < 0]
        if len(dn) < 2:
            sortino = 99.9
        else:
            dv = dn.std() * np.sqrt(12)
            sortino = (exc.mean() * 12) / dv if dv > 0.0001 else 0
        results.append({
            "Ticker": col,
            "Annualized Return %": round(ann, 2),
            "Total Return %": round((total - 1) * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
        })

    metrics_eq = pd.DataFrame(results)
    if metrics_eq.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Exclude benchmark from top-6 ranking
    non_bench = metrics_eq[metrics_eq["Ticker"] != BENCHMARK].copy()
    top6 = non_bench.sort_values("Annualized Return %", ascending=False).head(6)
    return top6, metrics_eq, normalized_eq


def render_sp500_section(months, selected_tf):
    st.markdown("---")
    st.subheader("🏦 S&P 500 Individual Stock Leaders")
    st.markdown(
        f"<p style='color:#94a3b8;font-size:13px;'>Top 6 performing S&P 500 companies over <b style='color:#38bdf8;'>{selected_tf}</b>, compared to VOO benchmark.</p>",
        unsafe_allow_html=True
    )

    with st.spinner("Fetching S&P 500 stock data…"):
        top6, metrics_eq, normalized_eq = fetch_sp500_top6(months)

    if top6.empty:
        st.warning("Could not load S&P 500 stock data for this timeframe.")
        return

    bench_row = metrics_eq[metrics_eq["Ticker"] == BENCHMARK]
    bench_data = bench_row.iloc[0] if not bench_row.empty else None

    # ── Circle cards ──────────────────────────────────────────────────────────
    cols_eq = st.columns(7)
    for idx, (_, row) in enumerate(top6.iterrows()):
        tk = row["Ticker"]
        bc = TOP6_COLORS[idx % len(TOP6_COLORS)]
        with cols_eq[idx]:
            st.markdown(f"""
            <div class="circle-card" style="border:3px solid {bc};">
                <div class="circle-ticker">{tk}</div>
                <div class="circle-meta">S&P 500 · Stock</div>
                <div class="circle-return" style="color:{bc};">+{row['Annualized Return %']}%</div>
                <div class="circle-metrics">
                    Sharpe: {row['Sharpe Ratio']}<br>
                    Sortino: {row['Sortino Ratio']:.1f}
                </div>
            </div>""", unsafe_allow_html=True)

    if bench_data is not None:
        with cols_eq[6]:
            st.markdown(f"""
            <div class="circle-card" style="border:3px solid #ef4444;background:linear-gradient(135deg,#311010,#0f172a);">
                <div class="circle-ticker">{BENCHMARK}</div>
                <div class="circle-meta">S&P 500 · Vanguard</div>
                <div class="circle-return" style="color:#ef4444;">+{bench_data['Annualized Return %']}%</div>
                <div class="circle-metrics">
                    Sharpe: {bench_data['Sharpe Ratio']}<br>
                    Sortino: {bench_data['Sortino Ratio']:.1f}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter: top-50 by annualized return — Sharpe vs Return ───────────────
    st.subheader("🎯 S&P 500 Top-50 Risk-Adjusted Landscape (Sharpe vs Annualized Return)")

    top50 = metrics_eq[metrics_eq["Ticker"] != BENCHMARK].nlargest(50, "Annualized Return %")
    fig_eq_s = go.Figure()

    top6_tickers = set(top6["Ticker"].tolist())
    for _, r in top50.iterrows():
        is_top6 = r["Ticker"] in top6_tickers
        fig_eq_s.add_trace(go.Scatter(
            x=[r["Annualized Return %"]], y=[r["Sharpe Ratio"]],
            mode="markers+text",
            name=r["Ticker"] if is_top6 else "Other Top-50",
            text=[r["Ticker"]],
            textposition="top center",
            textfont=dict(size=9, color="#f59e0b" if is_top6 else "#94a3b8"),
            marker=dict(
                size=14 if is_top6 else 9,
                color="#f59e0b" if is_top6 else "#64748b",
                opacity=0.9,
                line=dict(width=1, color="#0f172a")
            ),
            showlegend=is_top6,
        ))

    if bench_data is not None:
        fig_eq_s.add_trace(go.Scatter(
            x=[bench_data["Annualized Return %"]], y=[bench_data["Sharpe Ratio"]],
            mode="markers+text", name=BENCHMARK_LABEL,
            text=[BENCHMARK], textposition="top center",
            textfont=dict(size=10, color="#ef4444", family="Arial Black"),
            marker=dict(size=14, color="#ef4444", symbol="star", line=dict(width=1, color="#fff")),
        ))

    fig_eq_s.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        height=480, margin=dict(l=20, r=20, t=15, b=20), hovermode="closest",
        legend=dict(
            orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
            font=dict(color="#f8fafc", size=11), bgcolor="rgba(15,23,42,0.85)",
            bordercolor="#475569", borderwidth=1,
        ),
        xaxis=dict(title=dict(text="Annualized Return (%)", font=dict(color="#94a3b8")),
                   gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   zeroline=True, zerolinecolor="#475569"),
        yaxis=dict(title=dict(text="Sharpe Ratio", font=dict(color="#94a3b8")),
                   gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   zeroline=True, zerolinecolor="#475569"),
    )
    st.plotly_chart(fig_eq_s, use_container_width=True)

    # ── Performance Tracking: top-6 vs VOO ────────────────────────────────────
    st.subheader("📈 S&P 500 Top-6 Performance Tracking Matrix ($1 Base Allocation vs VOO)")

    chart_tks = list(top6["Ticker"]) + [BENCHMARK]
    chart_tks = [t for t in chart_tks if t in normalized_eq.columns]
    plot_eq   = normalized_eq[chart_tks].copy()

    fig_eq_l = go.Figure()
    for idx, tk in enumerate(chart_tks):
        is_b = (tk == BENCHMARK)
        lbl  = BENCHMARK_LABEL if is_b else tk
        fig_eq_l.add_trace(go.Scatter(
            x=plot_eq.index, y=plot_eq[tk].round(3), name=lbl,
            line=dict(
                color="#ef4444" if is_b else TOP6_COLORS[idx % len(TOP6_COLORS)],
                width=3.5 if is_b else 2.2,
                dash="dash" if is_b else "solid"
            )
        ))

    fig_eq_l.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        margin=dict(l=20, r=20, t=15, b=20), height=480, hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
            font=dict(color="#f8fafc", size=11), bgcolor="rgba(15,23,42,0.85)",
            bordercolor="#475569", borderwidth=1,
        ),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   title=dict(text="Growth Multiplier ($)", font=dict(color="#94a3b8")))
    )
    st.plotly_chart(fig_eq_l, use_container_width=True)


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
    _render_table("engine-table", col_labels, rows)


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
    _render_table("explorer-table", col_labels, rows)


# ── RENDER TIMEFRAMES TAB ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_all_timeframes(tickers_tuple):
    """Fetch price data for all timeframes and all tickers at once (longest window)."""
    tickers = list(tickers_tuple)
    end_date   = datetime.today()
    start_date = end_date - relativedelta(months=73)  # 6 years + 1 month buffer
    unique_tickers = list(set(tickers + [BENCHMARK]))
    try:
        raw = yf.download(unique_tickers, start=start_date, end=end_date,
                          progress=False, auto_adjust=True, threads=True)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    df_close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) \
               else pd.DataFrame({unique_tickers[0]: raw['Close']})
    monthly = df_close.resample('ME').last().ffill().bfill()
    return monthly


def compute_tf_metrics(monthly, months):
    """Compute annualized return for all tickers over a given month window."""
    end_date = datetime.today()
    window_start = end_date - relativedelta(months=months)
    subset = monthly[monthly.index >= window_start]
    if subset.empty or len(subset) < 2:
        return {}
    norm = subset / subset.iloc[0]
    n_years = months / 12.0
    out = {}
    for col in norm.columns:
        total = norm[col].iloc[-1]
        ann   = (total ** (1 / n_years) - 1) * 100 if total > 0 and n_years > 0 else 0
        out[col] = round(ann, 2)
    return out


def render_timeframes(metadata):
    st.subheader("📅 Index Fund Returns Across All Timeframes")
    st.markdown(
        "<p style='color:#94a3b8;font-size:13px;'>Annualized return (% p.a.) for all 192 index funds "
        "across every time horizon — from 3 months to 6 years. "
        "Green = outperforms VOO, Red = underperforms, sorted by 3-Year return by default.</p>",
        unsafe_allow_html=True
    )

    tickers = metadata["Ticker"].dropna().tolist()

    with st.spinner("Loading full multi-timeframe dataset (this may take ~30 s)…"):
        monthly = fetch_all_timeframes(tuple(tickers))

    if monthly.empty:
        st.error("Could not load price data for timeframes.")
        return

    tf_list = list(TIMEFRAME_MAP.keys())  # 6Y → 3M
    tf_data = {}
    for label, mo in TIMEFRAME_MAP.items():
        tf_data[label] = compute_tf_metrics(monthly, mo)

    # Build table rows
    rows = []
    for _, m in metadata.iterrows():
        tk = m["Ticker"]
        if pd.isna(tk):
            continue
        row = {
            "Ticker":     tk,
            "Sub-Sector": m.get("Sub_Sector", ""),
            "Category":   m.get("Category", ""),
            "Issuer":     m.get("Issuer", ""),
        }
        for label in tf_list:
            row[label] = tf_data[label].get(tk, None)
        rows.append(row)

    df_tf = pd.DataFrame(rows)

    # Benchmark row for comparison
    bench_vals = {label: tf_data[label].get(BENCHMARK, None) for label in tf_list}

    # ── Heatmap visualisation ──────────────────────────────────────────────────
    st.markdown("#### 🌡️ Annualized Return Heatmap (All Funds × All Timeframes)")

    # Sort by 3-Year default, put benchmark first
    sort_col = "3 Years"
    bench_df    = df_tf[df_tf["Ticker"] == BENCHMARK].copy()
    non_bench_df = df_tf[df_tf["Ticker"] != BENCHMARK].copy()
    non_bench_df = non_bench_df.sort_values(sort_col, ascending=False)
    df_sorted = pd.concat([bench_df, non_bench_df], ignore_index=True)

    # Build Plotly heatmap
    z_vals  = df_sorted[tf_list].values.astype(float)
    y_labels = [
        f"{r['Ticker']} — {r['Sub-Sector'][:28]}" if pd.notna(r.get('Sub-Sector')) and r.get('Sub-Sector') else r['Ticker']
        for _, r in df_sorted.iterrows()
    ]

    fig_hm = go.Figure(go.Heatmap(
        z=z_vals,
        x=tf_list,
        y=y_labels,
        colorscale=[
            [0.0,  "#7f1d1d"],
            [0.25, "#ef4444"],
            [0.45, "#94a3b8"],
            [0.55, "#94a3b8"],
            [0.75, "#10b981"],
            [1.0,  "#064e3b"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        hovertemplate="<b>%{y}</b><br>%{x}: <b>%{z:.2f}%</b><extra></extra>",
        colorbar=dict(
            title=dict(text="Ann. Return %", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
            thickness=14,
        ),
    ))
    fig_hm.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=max(600, len(df_sorted) * 22),
        margin=dict(l=280, r=80, t=20, b=60),
        xaxis=dict(tickfont=dict(color="#94a3b8", size=12), side="top"),
        yaxis=dict(tickfont=dict(color="#f8fafc", size=10), autorange="reversed"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Bar chart: select a single timeframe ──────────────────────────────────
    st.markdown("#### 📊 Top Performers by Timeframe")
    sel_tf = st.selectbox("Select timeframe to rank:", tf_list, index=2, key="tf_rank_select")
    top_n  = st.slider("Show top N funds:", 10, 50, 20, key="tf_top_n")

    tf_ranked = df_tf[df_tf["Ticker"] != BENCHMARK].dropna(subset=[sel_tf])
    tf_ranked = tf_ranked.sort_values(sel_tf, ascending=False).head(top_n)
    bench_val = bench_vals.get(sel_tf)

    fig_bar = go.Figure()
    colors_bar = [
        "#10b981" if v >= (bench_val or 0) else "#ef4444"
        for v in tf_ranked[sel_tf]
    ]
    fig_bar.add_trace(go.Bar(
        x=tf_ranked["Ticker"],
        y=tf_ranked[sel_tf],
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in tf_ranked[sel_tf]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Ann. Return: %{y:.2f}%<extra></extra>",
    ))
    if bench_val is not None:
        fig_bar.add_hline(
            y=bench_val, line_dash="dash", line_color="#ef4444",
            annotation_text=f"VOO: {bench_val:.1f}%",
            annotation_font_color="#ef4444",
        )
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        height=420, margin=dict(l=20, r=20, t=20, b=60),
        xaxis=dict(tickfont=dict(color="#94a3b8"), tickangle=-45),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#94a3b8"),
                   title=dict(text="Annualized Return (%)", font=dict(color="#94a3b8"))),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Raw data table ─────────────────────────────────────────────────────────
    with st.expander("📋 Full Timeframe Data Table"):
        display_cols = ["Ticker", "Sub-Sector", "Category"] + tf_list
        df_show = df_sorted[display_cols].copy()
        for col in tf_list:
            df_show[col] = df_show[col].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
        st.dataframe(df_show, use_container_width=True, hide_index=True)


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

    tab_home, tab_metrics, tab_explorer, tab_timeframes = st.tabs([
        "🏠 Dashboard Performance Analysis",
        "📋 Full Metric Engine View",
        "🔍 Database Catalog Reference",
        "📅 Return on Timeframes",
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
    with tab_timeframes:
        render_timeframes(metadata)

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
