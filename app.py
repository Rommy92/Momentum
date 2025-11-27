import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
from ta.momentum import RSIIndicator

# -------------- PAGE CONFIG ------------------
st.set_page_config(page_title="Tech Leadership Monitor", layout="wide")


# -------------- TICKER STATUS + SESSION LOGIC ------------------


def get_ticker_status(symbol: str, allow_single: bool = False):
    """
    Return (mode, price, change, change_pct, arrow) for a ticker.

    mode: 'green', 'red', 'neutral'
    arrow: ▲ / ▼ / ▶
    """
    try:
        hist = yf.Ticker(symbol).history(period="2d")
        closes = hist["Close"].dropna()

        # No data at all
        if len(closes) == 0:
            return "neutral", None, None, None, "▶"

        # Only one close – can happen with futures
        if len(closes) == 1:
            if not allow_single:
                return "neutral", None, None, None, "▶"
            price = float(closes.iloc[-1])
            return "neutral", price, 0.0, 0.0, "▶"

        # Normal 2-day case
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


def is_regular_trading_hours():
    """
    US market cash session: Mon–Fri, 09:30–16:00 US/Eastern.
    Outside of this, we treat it as 'futures hours'.
    """
    now_et = pd.Timestamp.now(tz="US/Eastern")
    if now_et.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    t = now_et.time()
    start = dt.time(9, 30)
    end = dt.time(16, 0)
    return start <= t <= end


# Fetch both QQQ and NQ futures, then choose which drives theme/header
qqq_mode, qqq_price, qqq_change, qqq_change_pct, qqq_arrow = get_ticker_status("QQQ")
fut_mode, fut_price, fut_change, fut_change_pct, fut_arrow = get_ticker_status(
    "NQ=F", allow_single=True
)

# Decide which one is "active"
if is_regular_trading_hours() or fut_price is None:
    # Cash hours OR futures unavailable → use QQQ
    active_label = "QQQ"
    active_mode = qqq_mode
    active_price = qqq_price
    active_change_pct = qqq_change_pct
    active_arrow = qqq_arrow
else:
    # Futures hours → NQ=F powers theme + header, but we call it QQQ Futures
    active_label = "QQQ Futures"
    active_mode = fut_mode
    active_price = fut_price
    active_change_pct = fut_change_pct
    active_arrow = fut_arrow


# Accent color based on ACTIVE driver
if active_mode == "green":
    accent = "#76B900"   # NVIDIA green
elif active_mode == "red":
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

if active_price is not None and active_change_pct is not None:
    st.subheader(
        f"{active_label} {active_arrow} {active_price:.2f} ({active_change_pct:+.2f}%)"
    )
else:
    st.subheader(f"{active_label} data unavailable — default neutral theme")

st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")


# -------------- TICKERS ------------------

TOP_TECH_TICKERS = [
    "MSFT", "AMZN", "GOOG", "NVDA", "META",
    "TSM", "AVGO", "ORCL", "CRM",
    "AMD", "NOW", "MU", "SNOW", "PLTR",
    "ANET", "CRWD", "PANW", "NET", "DDOG",
    "MDB", "MRVL", "IBM", "AMKR", "SMCI",
    "AXON", "ISRG", "TSLA"
]

# Full Nasdaq-100 list (as of late 2025)
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


def rsi_zone_text(rsi_val: float) -> str:
    """Return text like '73.9 – Overbought'."""
    if rsi_val < 30:
        zone = "Oversold"
    elif rsi_val < 50:
        zone = "Watch"
    elif rsi_val < 70:
        zone = "Trend"
    else:
        zone = "Overbought"
    return f"{rsi_val:.1f} – {zone}"


def rsi_zone_style(val):
    """Colour RSI Zone cell based on numeric RSI inside the text."""
    if val is None:
        return ""
    try:
        num_str = str(val).split()[0]  # "73.9 – Overbought" -> "73.9"
        rsi = float(num_str)
    except Exception:
        return ""

    if rsi < 30:
        return "color: #22c55e; font-weight: 600;"  # green
    if rsi < 50:
        return "color: #eab308; font-weight: 600;"  # yellow
    if rsi < 70:
        return "color: #3b82f6; font-weight: 600;"  # blue
    return "color: #ef4444; font-weight: 600;"       # red


# --- heatmap helpers (pure CSS, no matplotlib) ---

RED = (239, 68, 68)
BLACK = (0, 0, 0)
GREEN = (34, 197, 94)


def _blend(c_from, c_to, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(
        int(round(cf + (ct - cf) * t))
        for cf, ct in zip(c_from, c_to)
    )


def _rgb_css(c):
    return f"background-color: rgb({c[0]}, {c[1]}, {c[2]});"


def color_tripolar(v, vmin, vmax):
    """
    Red -> Black -> Green around 0.
    vmin <= v <= vmax, usually vmin<0<vmax.
    """
    if pd.isna(v) or vmin is None or vmax is None or vmin == vmax:
        return ""
    v = float(v)
    mid = 0.0

    if v < mid:
        if vmin >= mid:
            return ""
        t = (v - mid) / (vmin - mid)  # in [0,1]
        col = _blend(BLACK, RED, t)
    else:
        if vmax <= mid:
            return ""
        t = (v - mid) / (vmax - mid)  # in [0,1]
        col = _blend(BLACK, GREEN, t)

    return _rgb_css(col)


def color_bipolar(v, vmin, vmax):
    """
    Red -> Green, used for % from 52w High (vmin negative, vmax ~0).
    """
    if pd.isna(v) or vmin is None or vmax is None or vmin == vmax:
        return ""
    v = float(v)
    t = (v - vmin) / (vmax - vmin)  # maps vmin->0, vmax->1
    col = _blend(RED, GREEN, t)
    return _rgb_css(col)


# -------------- DATA FETCH ------------------


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

            rsi_zone_str = rsi_zone_text(rsi_val)

            value_signal = get_value_momentum_signal(
                rsi=rsi_val,
                pct_from_high=pct_from_52wk,
                pct_1m=pct_1m,
                fpe=fpe
            )

            rows.append(
                {
                    "Ticker": ticker,
                    "Price": price,
                    "% 5D": pct_5d,
                    "% 1M": pct_1m,
                    "% from 52w High": pct_from_52wk,
                    "RSI Zone": rsi_zone_str,
                    "Value Signal": value_signal,
                    "P/E": pe,
                    "Fwd P/E": fpe,
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)


# -------------- COMMON COLUMN CONFIG ------------------

BASE_COLUMN_CONFIG = {
    col: st.column_config.Column(width="fit")  # auto-size
    for col in [
        "Price",
        "% 5D",
        "% 1M",
        "% from 52w High",
        "RSI Zone",
        "Value Signal",
        "P/E",
        "Fwd P/E",
    ]
}


def build_column_config(columns):
    """Return column_config dict for given columns."""
    cfg = {}
    for col in columns:
        if col in BASE_COLUMN_CONFIG:
            cfg[col] = BASE_COLUMN_CONFIG[col]
        else:
            cfg[col] = st.column_config.Column(width="fit")
    return cfg


# -------------- TABLE 1: TECH LEADERSHIP MONITOR ------------------

with st.spinner("📡 Fetching data for Tech Leadership Monitor..."):
    df = get_stock_summary(TOP_TECH_TICKERS)

if not df.empty:
    df = df.set_index("Ticker")
    df_display = df.copy()

    format_dict = {
        "Price": "${:,.2f}",
        "% 5D": "{:.1f}%",
        "% 1M": "{:.1f}%",
        "% from 52w High": "{:.1f}%",
        "P/E": "{:.1f}",
        "Fwd P/E": "{:.1f}",
    }

    styled = df_display.style.format(format_dict, na_rep="–")

    # Heatmaps
    pct_cols = ["% 5D", "% 1M"]
    dist_col = "% from 52w High"

    for col in pct_cols:
        if df_display[col].notna().any():
            vmin = df_display[col].min()
            vmax = df_display[col].max()
            styled = styled.apply(
                lambda s, vmin=vmin, vmax=vmax: [
                    color_tripolar(v, vmin, vmax) for v in s
                ],
                subset=[col],
                axis=0,
            )

    if df_display[dist_col].notna().any():
        vmin = df_display[dist_col].min()
        vmax = 0.0
        styled = styled.apply(
            lambda s, vmin=vmin, vmax=vmax: [
                color_bipolar(v, vmin, vmax) for v in s
            ],
            subset=[dist_col],
            axis=0,
        )

    # Center ALL cells + headers
    styled = styled.set_table_styles(
        [
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ],
        overwrite=False,
    )

    styled = styled.applymap(rsi_zone_style, subset=["RSI Zone"])

    column_config = build_column_config(df_display.columns)

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        column_config=column_config,
    )
else:
    st.write("No data loaded.")


# -------------- TABLE 2: NASDAQ-100 DEEP DRAWDOWN ------------------

st.markdown("---")
st.markdown(
    "<h1 style='text-align:left; text-shadow:0 0 8px #76B900;'>"
    "NASDAQ-100 DEEP DRAWDOWN TABLE</h1>",
    unsafe_allow_html=True,
)

with st.spinner("📡 Fetching Nasdaq-100 data..."):
    df_ndx = get_stock_summary(NASDAQ100_TICKERS)

if not df_ndx.empty:
    df_ndx = df_ndx.sort_values("% from 52w High")
    df_ndx = df_ndx.set_index("Ticker")
    df_ndx_display = df_ndx.copy()

    ndx_format_dict = {
        "Price": "${:,.2f}",
        "% 5D": "{:.1f}%",
        "% 1M": "{:.1f}%",
        "% from 52w High": "{:.1f}%",
        "P/E": "{:.1f}",
        "Fwd P/E": "{:.1f}",
    }

    styled_ndx = df_ndx_display.style.format(ndx_format_dict, na_rep="–")

    ndx_pct_cols = ["% 5D", "% 1M"]
    ndx_dist_col = "% from 52w High"

    for col in ndx_pct_cols:
        if df_ndx_display[col].notna().any():
            vmin = df_ndx_display[col].min()
            vmax = df_ndx_display[col].max()
            styled_ndx = styled_ndx.apply(
                lambda s, vmin=vmin, vmax=vmax: [
                    color_tripolar(v, vmin, vmax) for v in s
                ],
                subset=[col],
                axis=0,
            )

    if df_ndx_display[ndx_dist_col].notna().any():
        vmin = df_ndx_display[ndx_dist_col].min()
        vmax = 0.0
        styled_ndx = styled_ndx.apply(
            lambda s, vmin=vmin, vmax=vmax: [
                color_bipolar(v, vmin, vmax) for v in s
            ],
            subset=[ndx_dist_col],
            axis=0,
        )

    styled_ndx = styled_ndx.set_table_styles(
        [
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ],
        overwrite=False,
    )

    styled_ndx = styled_ndx.applymap(rsi_zone_style, subset=["RSI Zone"])

    ndx_column_config = build_column_config(df_ndx_display.columns)

    st.dataframe(
        styled_ndx,
        use_container_width=True,
        height=600,
        column_config=ndx_column_config,
    )
else:
    st.write("No Nasdaq-100 data loaded.")


# -------------- HOW TO READ THE SIGNALS ------------------

st.markdown("---")
st.markdown(
    """
**Value Signal (combined value + momentum)**  
- 💚 **Deep value pullback** – Big drawdown vs 52-week high, low forward P/E, weak RSI.  
- 🟡 **Value watch** – Decent pullback, valuation reasonable but not screaming.  
- 🔵 **Momentum trend** – Positive 1-month performance with RSI in 50–70 zone.  
- 🔴 **Hot / extended** – Near highs and/or expensive forward P/E, or overbought RSI.  
- ⚪ **Neutral** – No strong edge from value or momentum.
"""
)

