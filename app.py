import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

# -------------- PAGE CONFIG ------------------
st.set_page_config(page_title="Tech Leadership Monitor", layout="wide")


# -------------- QQQ MODE + THEME LOGIC ------------------

def get_qqq_status():
    """
    Fetch last 2 days of QQQ and decide:
    - mode: 'green', 'red', 'neutral'
    - price: latest close
    - change: absolute change vs previous close
    - change_pct: percentage change
    - arrow: '▲' / '▼' / '▶'
    """
    try:
        hist = yf.Ticker("QQQ").history(period="2d")
        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return "neutral", None, None, None, "▶"

        prev_price = float(closes.iloc[-2])
        price = float(closes.iloc[-1])
        change = price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0.0

        if change > 0:
            mode = "green"
            arrow = "▲"
        elif change < 0:
            mode = "red"
            arrow = "▼"
        else:
            mode = "neutral"
            arrow = "▶"

        return mode, price, change, change_pct, arrow
    except Exception:
        return "neutral", None, None, None, "▶"


qqq_mode, qqq_price, qqq_change, qqq_change_pct, qqq_arrow = get_qqq_status()

# Accent color based on QQQ (green = NVIDIA green)
if qqq_mode == "green":
    accent = "#76B900"   # NVIDIA green
elif qqq_mode == "red":
    accent = "#ef4444"   # soft red
else:
    accent = "#0ea5e9"   # cyan/blue


# -------------- CYBERPUNK CSS ------------------

cyberpunk_css = f"""
<style>
/* App + sidebar background */
[data-testid="stAppViewContainer"] {{
    background-color: #000000 !important;
    color: #eeeeee !important;
}}

[data-testid="stSidebar"] {{
    background-color: #000000 !important;
    color: #eeeeee !important;
    border-right: 1px solid {accent}33 !important;
}}

/* Global text */
html, body, [class*="css"] {{
    color: #eeeeee !important;
    background-color: #000000 !important;
}}

/* Headings with soft glow */
h1, h2 {{
    color: {accent} !important;
    text-shadow: 0 0 4px {accent}, 0 0 10px {accent};
    animation: neonPulse 3s ease-in-out infinite;
}}

h3, h4 {{
    color: {accent} !important;
}}

@keyframes neonPulse {{
    0% {{
        text-shadow: 0 0 4px {accent}, 0 0 8px {accent};
    }}
    50% {{
        text-shadow: 0 0 12px {accent}, 0 0 22px {accent};
    }}
    100% {{
        text-shadow: 0 0 4px {accent}, 0 0 8px {accent};
    }}
}}

/* Full-width content container */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    max-width: 100% !important;
}}

/* Dark styling for st.dataframe */
[data-testid="stDataFrame"] div[role="grid"] {{
    background-color: #050505 !important;
    color: #ffffff !important;
}}

[data-testid="stDataFrame"] div[role="columnheader"] {{
    background-color: #101010 !important;
    color: {accent} !important;
    border-bottom: 1px solid {accent}77 !important;
}}

[data-testid="stDataFrame"] div[role="cell"] {{
    border-bottom: 1px solid #222222 !important;
}}
</style>
"""
st.markdown(cyberpunk_css, unsafe_allow_html=True)


# -------------- TITLE + SUBHEADER ------------------

st.title("Tech Leadership Monitor")

if qqq_price is not None and qqq_change_pct is not None:
    st.subheader(f"QQQ {qqq_arrow} {qqq_price:.2f} ({qqq_change_pct:+.2f}%)")
else:
    st.subheader("QQQ data unavailable — default neutral theme")

st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")


# -------------- TICKER LISTS ------------------

TOP_TECH_TICKERS = [
    "MSFT", "AMZN", "GOOG", "NVDA", "META",
    "TSM", "AVGO", "ORCL", "CRM",
    "AMD", "NOW", "MU", "SNOW", "PLTR",
    "ANET", "CRWD", "PANW", "NET", "DDOG",
    "MDB", "MRVL", "IBM", "AMKR", "SMCI",
    "AXON", "ISRG",
]

NASDAQ100_TICKERS = [
    "ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI",
    "AAPL", "AMAT", "APP", "ARM", "ASML", "AZN", "TEAM", "ADSK", "ADP",
    "AXON", "BKR", "BIIB", "BKNG", "AVGO", "CDNS", "CDW", "CHTR", "CTAS",
    "CSCO", "CCEP", "CTSH", "CMCSA", "CEG", "CPRT", "CSGP", "COST", "CRWD",
    "CSX", "DDOG", "DXCM", "FANG", "DASH", "EA", "EXC", "FAST", "FTNT",
    "GEHC", "GILD", "GFS", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP",
    "KLAC", "KHC", "LRCX", "LIN", "LULU", "MAR", "MRVL", "MELI", "META",
    "MCHP", "MU", "MSFT", "MSTR", "MDLZ", "MNST", "NFLX", "NVDA", "NXPI",
    "ORLY", "ODFL", "ON", "PCAR", "PLTR", "PANW", "PAYX", "PYPL", "PDD",
    "PEP", "QCOM", "REGN", "ROP", "ROST", "SHOP", "SOLS", "SBUX", "SNPS",
    "TMUS", "TTWO", "TSLA", "TXN", "TRI", "TTD", "VRSK", "VRTX", "WBD",
    "WDAY", "XEL", "ZS",
]


# -------------- HELPERS ------------------

def get_value_momentum_signal(rsi, pct_from_high, pct_1m, fpe):
    if rsi is None or pct_from_high is None:
        return "❔ Check data"

    if rsi < 35 and pct_from_high <= -30 and (fpe is not None and fpe <= 30):
        return "💚 Deep value pullback"

    if rsi < 50 and pct_from_high <= -15 and (fpe is not None and fpe <= 35):
        return "🟡 Value watch"

    if 50 <= rsi <= 70 and (pct_1m is not None and pct_1m > 0):
        return "🔵 Momentum trend"

    if rsi > 70 or (pct_from_high >= -5 and (fpe is not None and fpe >= 45)):
        return "🔴 Hot / extended"

    return "⚪ Neutral"


# --- heatmap helpers (pure CSS, no matplotlib) ---

RED = (239, 68, 68)
BLACK = (0, 0, 0)
GREEN = (34, 197, 94)


def _blend(c_from, c_to, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(cf + (ct - cf) * t)) for cf, ct in zip(c_from, c_to))


def _rgb_css(c):
    return f"background-color: rgb({c[0]}, {c[1]}, {c[2]});"


def color_tripolar(v, vmin, vmax):
    """Red -> Black -> Green around 0 (for % 5D / % 1M)."""
    if pd.isna(v) or vmin is None or vmax is None or vmin == vmax:
        return ""
    v = float(v)
    mid = 0.0

    if v < mid:
        if vmin >= mid:
            return ""
        t = (v - mid) / (vmin - mid)
        col = _blend(BLACK, RED, t)
    else:
        if vmax <= mid:
            return ""
        t = (v - mid) / (vmax - mid)
        col = _blend(BLACK, GREEN, t)

    return _rgb_css(col)


def color_bipolar(v, vmin, vmax):
    """Red -> Green, used for % from 52w High (vmin negative, vmax ~0)."""
    if pd.isna(v) or vmin is None or vmax is None or vmin == vmax:
        return ""
    v = float(v)
    t = (v - vmin) / (vmax - vmin)
    col = _blend(RED, GREEN, t)
    return _rgb_css(col)


def rsi_zone_style(val):
    """Colour the RSI Zone text itself."""
    if pd.isna(val):
        return ""
    text = str(val)
    if "Oversold" in text:
        return "color: #22c55e; font-weight: 600;"   # green
    if "Trend" in text:
        return "color: #3b82f6; font-weight: 600;"   # blue
    if "Watch" in text:
        return "color: #facc15; font-weight: 600;"   # yellow
    if "Overbought" in text:
        return "color: #ef4444; font-weight: 600;"   # red
    return ""


@st.cache_data(ttl=600)
def get_stock_summary(tickers):
    rows = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty or "Close" not in hist.columns:
                continue

            close = hist["Close"].dropna()
            if len(close) < 10:
                continue

            price = float(close.iloc[-1])

            pct_5d = (
                round((price - float(close.iloc[-6])) / float(close.iloc[-6]) * 100, 2)
                if len(close) >= 6 else None
            )
            pct_1m = (
                round((price - float(close.iloc[-22])) / float(close.iloc[-22]) * 100, 2)
                if len(close) >= 22 else None
            )

            high_52wk = float(close.max())
            pct_from_52wk = round((price - high_52wk) / high_52wk * 100, 2)

            rsi_series = RSIIndicator(close=close).rsi()
            rsi_val = float(round(rsi_series.iloc[-1], 2))

            # Fundamentals
            try:
                info = stock.get_info()
            except Exception:
                info = {}

            pe = info.get("trailingPE", None)
            try:
                pe = float(pe)
            except Exception:
                pe = None

            # Forward EPS (eps_trend + fallback to forwardEps)
            fpe = None
            forward_eps = None
            try:
                eps_trend = stock.get_eps_trend()
                if eps_trend is not None and not eps_trend.empty:
                    idx = None
                    for candidate in ["+1y", "0y"]:
                        if candidate in eps_trend.index:
                            idx = candidate
                            break
                    if idx is not None and "current" in eps_trend.columns:
                        val = eps_trend.loc[idx, "current"]
                        if pd.notna(val):
                            forward_eps = float(val)
            except Exception:
                forward_eps = None

            if forward_eps is None:
                try:
                    fe = info.get("forwardEps", None)
                    forward_eps = float(fe) if fe is not None else None
                except Exception:
                    forward_eps = None

            if forward_eps and forward_eps > 0:
                try:
                    fpe = round(price / forward_eps, 2)
                except ZeroDivisionError:
                    fpe = None

            # RSI zone + emoji, with numeric value merged
            if rsi_val < 30:
                zone_label = "Oversold"
                emoji = "💚"
            elif rsi_val < 50:
                zone_label = "Watch"
                emoji = "🟡"
            elif rsi_val < 70:
                zone_label = "Trend"
                emoji = "🔵"
            else:
                zone_label = "Overbought"
                emoji = "🔴"

            rsi_zone_display = f"{emoji} {zone_label} ({rsi_val:.1f})"

            value_signal = get_value_momentum_signal(
                rsi=rsi_val,
                pct_from_high=pct_from_52wk,
                pct_1m=pct_1m,
                fpe=fpe,
            )

            rows.append(
                {
                    "Ticker": ticker,
                    "Price": price,
                    "% 5D": pct_5d,
                    "% 1M": pct_1m,
                    "% from 52w High": pct_from_52wk,
                    # RSI kept internal (not displayed)
                    "RSI": rsi_val,
                    "RSI Zone": rsi_zone_display,
                    "Value Signal": value_signal,
                    "P/E": pe,
                    "Fwd P/E": fpe,
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)


def make_styled_table(df: pd.DataFrame):
    """
    Return (styled, column_config) so both tables share exact look.
    NOTE: df contains internal 'RSI' numeric column that we drop from display.
    """
    # Hide raw RSI column from the UI
    df_display = df.drop(columns=["RSI"])
    df_display = df_display.set_index("Ticker")

    format_dict = {
        "Price": "${:,.2f}",
        "% 5D": "{:.1f}%",
        "% 1M": "{:.1f}%",
        "% from 52w High": "{:.1f}%",
        "P/E": "{:.1f}",
        "Fwd P/E": "{:.1f}",
    }

    styled = df_display.style.format(format_dict, na_rep="–")

    # --- Heatmaps use underlying numeric df (not df_display) ---
    pct_cols = ["% 5D", "% 1M"]
    dist_col = "% from 52w High"

    for col in pct_cols:
        if df[col].notna().any():
            vmin = df[col].min()
            vmax = df[col].max()
            styled = styled.apply(
                lambda s, vmin=vmin, vmax=vmax: [
                    color_tripolar(v, vmin, vmax) for v in s
                ],
                subset=[col],
                axis=0,
            )

    if df[dist_col].notna().any():
        vmin = df[dist_col].min()
        vmax = 0.0
        styled = styled.apply(
            lambda s, vmin=vmin, vmax=vmax: [
                color_bipolar(v, vmin, vmax) for v in s
            ],
            subset=[dist_col],
            axis=0,
        )

    # Right-align numeric columns, center headers
    numeric_cols = ["Price", "% 5D", "% 1M", "% from 52w High", "P/E", "Fwd P/E"]

    styled = (
        styled.set_table_styles(
            [{"selector": "th.col_heading", "props": [("text-align", "center")]}],
            overwrite=False,
        )
        .set_properties(subset=numeric_cols, **{"text-align": "right"})
    )

    # Colour RSI Zone text
    styled = styled.applymap(rsi_zone_style, subset=["RSI Zone"])

    # ---------- Column widths (tighter) ----------
    # - P/E, Fwd P/E      → very narrow
    # - Price, % 5D, % 1M → narrow
    # - % from 52w High   → medium
    # - RSI Zone          → 2/3 of previous
    # - Value Signal      → ~half of previous

column_config = {

    "Price": st.column_config.Column(width=60),
    "% 5D": st.column_config.Column(width=50),
    "% 1M": st.column_config.Column(width=50),
    "% from 52w High": st.column_config.Column(width=110),

    "RSI Zone": st.column_config.Column(width=120),
    "Value Signal": st.column_config.Column(width=150),

    "P/E": st.column_config.Column(width=35),
    "Fwd P/E": st.column_config.Column(width=35),
}
            # fallback for any future extra column
            column_config[col] = st.column_config.Column(width=140)

    return styled, column_config


# -------------- TABLE 1: TECH LEADERSHIP ------------------

with st.spinner("📡 Fetching tech leadership data..."):
    df_top = get_stock_summary(TOP_TECH_TICKERS)

if not df_top.empty:
    styled_top, col_cfg_top = make_styled_table(df_top)
    st.dataframe(styled_top, use_container_width=True, height=600, column_config=col_cfg_top)
else:
    st.write("No data loaded for tech tickers.")

st.markdown("---")
st.markdown("## NASDAQ-100 Deep Drawdown Table")


# -------------- TABLE 2: NASDAQ-100 ------------------

with st.spinner("📡 Fetching Nasdaq-100 data..."):
    df_ndx = get_stock_summary(NASDAQ100_TICKERS)

if not df_ndx.empty:
    df_ndx = df_ndx.sort_values("% from 52w High")  # deepest drawdown at top
    styled_ndx, col_cfg_ndx = make_styled_table(df_ndx)
    st.dataframe(styled_ndx, use_container_width=True, height=600, column_config=col_cfg_ndx)
else:
    st.write("No Nasdaq-100 data loaded.")
