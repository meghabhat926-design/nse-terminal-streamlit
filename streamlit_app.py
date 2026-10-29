"""
NSE Stock Dashboard — Streamlit frontend (free hosting on Streamlit Community Cloud)

Reuses the indicator + fundamentals functions from nse_analysis_api.py, so keep both
files in the same GitHub repo. No separate API server is needed for this dashboard.

New features added on top of your API:
  1. Valuation ratios (P/E, P/B, dividend yield)
  2. Relative strength vs Nifty 50 (1M / 3M / 6M) + beta
  3. Composite score (0-100) with a factor-by-factor breakdown
"""

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nse_analysis_api import (
    calculate_rsi,
    calculate_macd,
    calculate_moving_averages,
    get_technical_snapshot,
    get_fundamentals,
    safe_round,
)

st.set_page_config(page_title="NSE Stock Dashboard", page_icon="📈", layout="wide")


# ── helpers ────────────────────────────────────────────────────────────────

def to_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    return symbol if symbol.endswith(".NS") or symbol.startswith("^") else f"{symbol}.NS"


def fmt(value, suffix="", prefix=""):
    return "N/A" if value is None else f"{prefix}{value:,}{suffix}"


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker: str) -> dict:
    return get_fundamentals(yf.Ticker(ticker))


# ── NEW FEATURE 1: valuation ratios ────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_valuation(ticker: str) -> dict:
    """P/E, P/B and dividend yield, built from statements (avoids stock.info,
    which Yahoo often blocks on cloud hosts)."""
    out = {"pe": None, "pb": None, "dividend_yield_pct": None}
    stock = yf.Ticker(ticker)

    mcap = None
    try:
        fi = stock.fast_info
        mcap = fi.get("marketCap") if fi else None
    except Exception:
        pass

    try:  # P/E = market cap / trailing-4-quarter net income
        inc = stock.quarterly_income_stmt
        if inc is not None and "Net Income" in inc.index and mcap:
            ni = inc.loc["Net Income"].dropna()
            if len(ni) >= 4:
                ttm = ni.iloc[:4].sum()
                if ttm > 0:
                    out["pe"] = safe_round(mcap / ttm)
    except Exception:
        pass

    try:  # P/B = market cap / latest shareholders' equity
        bs = stock.quarterly_balance_sheet
        if bs is not None and mcap:
            for key in ("Stockholders Equity", "Common Stock Equity",
                        "Total Equity Gross Minority Interest"):
                if key in bs.index:
                    eq = bs.loc[key].dropna()
                    if len(eq) and eq.iloc[0] > 0:
                        out["pb"] = safe_round(mcap / eq.iloc[0])
                        break
    except Exception:
        pass

    try:  # dividend yield = dividends paid in last 12 months / price
        div = stock.dividends
        price = stock.fast_info.get("lastPrice") if stock.fast_info else None
        if div is not None and len(div) and price:
            cutoff = pd.Timestamp.now(tz=div.index.tz) - pd.DateOffset(years=1)
            paid = div[div.index >= cutoff].sum()
            out["dividend_yield_pct"] = safe_round(paid / price * 100)
    except Exception:
        pass

    return out


# ── NEW FEATURE 2: relative strength vs Nifty 50 + beta ───────────────────

def relative_strength(hist: pd.DataFrame, bench: pd.DataFrame) -> dict:
    out = {}
    for label, n in (("1M", 21), ("3M", 63), ("6M", 126)):
        if len(hist) > n and len(bench) > n:
            s = hist["Close"].iloc[-1] / hist["Close"].iloc[-n - 1] - 1
            b = bench["Close"].iloc[-1] / bench["Close"].iloc[-n - 1] - 1
            out[label] = {
                "stock": round(s * 100, 2),
                "nifty": round(b * 100, 2),
                "excess": round((s - b) * 100, 2),
            }
    return out


def calc_beta(hist: pd.DataFrame, bench: pd.DataFrame):
    joined = pd.concat(
        [hist["Close"].pct_change(), bench["Close"].pct_change()], axis=1, join="inner"
    ).dropna().tail(252)
    if len(joined) < 60:
        return None
    cov = joined.iloc[:, 0].cov(joined.iloc[:, 1])
    var = joined.iloc[:, 1].var()
    return safe_round(cov / var) if var else None


# ── NEW FEATURE 3: composite score ─────────────────────────────────────────

def composite_score(tech: dict, fund: dict, val: dict, rel: dict):
    parts = []

    def add(name, pts, mx):
        parts.append({"Factor": name, "Points": pts, "Max": mx})

    add("Trend (MA50 vs MA200)", 20 if tech.get("ma_trend") == "bullish" else 0, 20)

    mom = 0
    if (tech.get("macd_histogram") or 0) > 0:
        mom += 10
    rsi = tech.get("rsi_14")
    if rsi is not None:
        if 40 <= rsi <= 70:
            mom += 10
        elif rsi < 30:
            mom += 5  # oversold: possible bounce
    add("Momentum (MACD + RSI)", mom, 20)

    adx = tech.get("adx_14") or 0
    add("Trend strength (ADX)", 10 if adx > 25 else 5 if adx > 20 else 0, 10)

    ex = rel.get("3M", {}).get("excess")
    add("Relative strength vs Nifty (3M)", 15 if (ex is not None and ex > 0) else 0, 15)

    growth = 0
    if (fund.get("yoy_earnings_growth_pct") or 0) > 0:
        growth += 8
    if (fund.get("yoy_revenue_growth_pct") or 0) > 0:
        growth += 7
    add("Growth (earnings + revenue YoY)", growth, 15)

    quality = 0
    if (fund.get("roe_pct") or 0) > 15:
        quality += 5
    de = fund.get("debt_to_equity")
    if de is not None and de < 1:
        quality += 5
    add("Quality (ROE + debt)", quality, 10)

    pe = val.get("pe")
    if pe is not None and 0 < pe < 25:
        v = 10
    elif pe is not None and 25 <= pe < 40:
        v = 5
    else:
        v = 0
    add("Valuation (P/E)", v, 10)

    score = sum(p["Points"] for p in parts)
    label = "Bullish" if score >= 70 else "Neutral" if score >= 45 else "Weak"
    return score, label, pd.DataFrame(parts)


# ── charting ───────────────────────────────────────────────────────────────

def make_chart(hist: pd.DataFrame, months: int) -> go.Figure:
    close = hist["Close"]
    rsi = calculate_rsi(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)

    d = hist.tail(months * 21)
    idx = d.index

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=idx, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ma50.loc[idx], name="MA50", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ma200.loc[idx], name="MA200", line=dict(width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=idx, y=rsi.loc[idx], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    colors = ["#26a69a" if (v >= 0) else "#ef5350" for v in macd_hist.loc[idx]]
    fig.add_trace(go.Bar(x=idx, y=macd_hist.loc[idx], name="MACD hist", marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=idx, y=macd_line.loc[idx], name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=idx, y=signal_line.loc[idx], name="Signal"), row=3, col=1)

    fig.update_layout(
        height=780, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.04), margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────

st.title("📈 NSE Stock Dashboard")
st.caption("Free, Python-only dashboard. Data: Yahoo Finance (may be delayed). "
           "For education only — not investment advice.")

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("NSE symbol", value="RELIANCE", help="e.g. TCS, INFY, HDFCBANK")
    months = st.selectbox("Chart range (months)", [3, 6, 12, 24], index=1)

tab_analysis, tab_chart, tab_watch = st.tabs(["Analysis", "Chart", "Watchlist"])

ticker = to_ticker(symbol)
hist = load_history(ticker)

with tab_analysis:
    if hist.empty:
        st.error(f"No data found for {ticker}. Check the symbol.")
    else:
        bench = load_history("^NSEI")
        tech, ca_flag = get_technical_snapshot(hist)
        fund = load_fundamentals(ticker)
        val = load_valuation(ticker)
        rel = relative_strength(hist, bench)
        beta = calc_beta(hist, bench)
        score, label, breakdown = composite_score(tech, fund, val, rel)

        if ca_flag:
            st.warning("A >15% single-day price move was detected in the last 2 years — "
                       "moving averages and Bollinger figures may be distorted by a split "
                       "or demerger. Verify before trusting the technical signals.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Price (₹)", fmt(tech["current_price"]))
        c2.metric("Composite score", f"{score}/100", label)
        c3.metric("Beta vs Nifty", fmt(beta))

        st.subheader("Technical")
        t = st.columns(5)
        t[0].metric("RSI (14)", fmt(tech["rsi_14"]))
        t[1].metric("MACD hist", fmt(tech["macd_histogram"]))
        t[2].metric("MA trend", (tech["ma_trend"] or "N/A").title())
        t[3].metric("ADX (14)", fmt(tech["adx_14"]))
        t[4].metric("% of 52w high", fmt(tech["pct_of_52w_high"], "%"))

        st.subheader("Valuation")
        v = st.columns(3)
        v[0].metric("P/E (TTM)", fmt(val["pe"]))
        v[1].metric("P/B", fmt(val["pb"]))
        v[2].metric("Dividend yield", fmt(val["dividend_yield_pct"], "%"))

        st.subheader("Fundamentals")
        f = st.columns(5)
        f[0].metric("Earnings YoY", fmt(fund["yoy_earnings_growth_pct"], "%"))
        f[1].metric("Revenue YoY", fmt(fund["yoy_revenue_growth_pct"], "%"))
        f[2].metric("Net margin", fmt(fund["net_margin_pct"], "%"))
        f[3].metric("ROE", fmt(fund["roe_pct"], "%"))
        f[4].metric("Debt/Equity", fmt(fund["debt_to_equity"]))

        st.subheader("Relative strength vs Nifty 50")
        if rel:
            st.dataframe(
                pd.DataFrame(rel).T.rename(columns={
                    "stock": "Stock %", "nifty": "Nifty %", "excess": "Excess %"}),
                use_container_width=True,
            )
        else:
            st.info("Not enough history to compare against Nifty.")

        with st.expander("How the composite score is calculated"):
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
            st.caption("Score ≥70 Bullish, 45–69 Neutral, <45 Weak. A simple rules-based "
                       "screen, not a recommendation.")

with tab_chart:
    if hist.empty:
        st.error(f"No data found for {ticker}.")
    else:
        st.plotly_chart(make_chart(hist, months), use_container_width=True)

with tab_watch:
    st.write("Enter up to 10 symbols, comma-separated.")
    raw = st.text_area("Watchlist", value="RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK")
    symbols = [s.strip() for s in raw.split(",") if s.strip()][:10]
    rows = []
    with st.spinner("Loading watchlist..."):
        for s in symbols:
            tk = to_ticker(s)
            try:
                h = load_history(tk)
                if h.empty:
                    rows.append({"Symbol": tk, "Note": "No data"})
                    continue
                tc, flag = get_technical_snapshot(h)
                rows.append({
                    "Symbol": tk,
                    "Price (₹)": tc["current_price"],
                    "RSI": tc["rsi_14"],
                    "MACD hist": tc["macd_histogram"],
                    "Trend": tc["ma_trend"],
                    "% of 52w high": tc["pct_of_52w_high"],
                    "Corp. action?": "⚠️" if flag else "",
                })
            except Exception as e:
                rows.append({"Symbol": tk, "Note": str(e)[:60]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
