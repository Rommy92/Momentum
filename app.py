import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
from ta.momentum import RSIIndicator

# -------------- PAGE CONFIG ------------------
st.set_page_config(
    page_title="Global Tech & Macro Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Small hint to show sidebar exists
st.markdown(
    """
    <style>
    .sidebar-hint {
        position: fixed;
        top: 0.5rem;
        left: 0.5rem;
        z-index: 999;
        background: rgba(15,23,42,0.9);
        padding: 0.25rem 0.5rem;
        border-radius: 999px;
        font-size: 0.75rem;
        color: #a5f3fc;
        border: 1px solid #22c55e55;
    }
    @media (max-width: 768px) {
        .sidebar-hint { display: none; }
    }
    </style>
    <div class="sidebar-hint">🟢 Open filters in sidebar ⟵</div>
    """,
    unsafe_allow_html=True,
)


# -------------- REALTIME TICKER STATUS ------------------


@st.cache_data(ttl=60)
def get_ticker_status(symbol: str):
    """
    Realtime-like status.

    Priority 1: use Yahoo 'regularMarket*' fields from get_info()
    Priority 2: fall back to 2-day daily history + 1m intraday.

    Returns (mode, price, change, change_pct, arrow).
    """
    # --------- PATH 1: regularMarket* from get_info() ----------
    try:
        t = yf.Ticker(symbol)
        info = t.get_info() or {}

        price = info.get("regularMarketPrice", None)
        change = info.get("regularMarketChange", None)
        change_pct = info.get("regularMarketChangePercent", None)

        if price is not None and change_pct is not None:
            price = float(price)
            change_pct = float(change_pct)

            if change is not None:
                change = float(change)
            else:
                try:
                    prev_close = price / (1 + change_pct / 100.0)
                    change = price - prev_close
                except Exception:
                    change = 0.0

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
        pass

    # --------- PATH 2: 2d daily + 1m intraday ----------
    try:
        t = yf.Ticker(symbol)

        daily = t.history(period="2d")
        closes = daily.get("Close", pd.Series(dtype=float)).dropna()
        if len(closes) == 0:
            return "neutral", None, None, None, "▶"

        if len(closes) >= 2:
            prev_close = float(closes.iloc[-2])
        else:
            prev_close = float(closes.iloc[-1])

        intra = t.history(period="1d", interval="1m", prepost=True)
        intra_closes = intra.get("Close", pd.Series(dtype=float)).dropna()
        if len(intra_closes) > 0:
            price = float(intra_closes.iloc[-1])
        else:
            price = prev_close

        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0

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


@st.cache_data(ttl=300)
def get_market_state(symbol: str):
    """
    Determine if a market is Open/Closed based on Yahoo's marketState.
    If unknown, default to Closed.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.get_info()
        state = str(info.get("marketState", "")).upper()
    except Exception:
        state = ""

    if state in ("REGULAR", "PRE", "POST"):
        return "Open"
    if state == "CLOSED":
        return "Closed"
    return "Closed"


def is_regular_trading_hours():
    """
    US market cash session: Mon–Fri, 09:30–16:00 US/Eastern.
    """
    now_et = pd.Timestamp.now(tz="US/Eastern")
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    start = dt.time(9, 30)
    end = dt.time(16, 0)
    return start <= t <= end


# Decide QQQ vs NQ futures
qqq_mode, qqq_price, qqq_change, qqq_change_pct, qqq_arrow = get_ticker_status("QQQ")
fut_mode, fut_price, fut_change, fut_change_pct, fut_arrow = get_ticker_status("NQ=F")

if is_regular_trading_hours() or fut_price is None:
    active_label = "QQQ"
    active_symbol = "QQQ"
    active_mode = qqq_mode
    active_price = qqq_price
    active_change_pct = qqq_change_pct
    active_arrow = qqq_arrow
else:
    active_label = "QQQ Futures"
    active_symbol = "NQ=F"
    active_mode = fut_mode
    active_price = fut_price
    active_change_pct = fut_change_pct
    active_arrow = fut_arrow


# Accent color based on ACTIVE driver
if active_mode == "green":
    accent = "#76B900"
elif active_mode == "red":
    accent = "#ef4444"
else:
    accent = "#0ea5e9"


# -------------- CYBERPUNK CSS ------------------

cyberpunk_css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: #000000 !important;
    color: #eeeeee !important;
}}

[data-testid="stSidebar"] {{
    background-color: #000000 !important;
    color: #eeeeee !important;
    border-right: 1px solid {accent}33 !important;
}}

html, body, [class*="css"] {{
    color: #eeeeee !important;
    background-color: #000000 !important;
}}

h1, h2 {{
    color: {accent} !important;
    text-shadow: 0 0 4px {accent}, 0 0 10px {accent};
    animation: neonPulse 3s ease-in-out infinite;
    text-align: center;
}}

h3, h4 {{
    color: {accent} !important;
    text-align: center;
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

.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    max-width: 100% !important;
}}

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

/* Make checkbox labels green & bold (Focus Table especially) */
div.row-widget.stCheckbox label span {{
    color: #22c55e !important;
    font-weight: 600 !important;
}}
</style>
"""
st.markdown(cyberpunk_css, unsafe_allow_html=True)


# -------------- SIDEBAR: BUY-ZONE FILTER CONTROLS ------------------

with st.sidebar:
    st.header("Buy-Zone Filters")
    dd_required = st.slider(
        "Drawdown from 52w High (%)",
        min_value=0,
        max_value=80,
        value=25,
        step=1,
        help="Minimum discount from 52-week high required (move right for bigger discount).",
    )
    min_dd = -float(dd_required)
    max_fpe = st.slider(
        "Max Forward P/E",
        min_value=5,
        max_value=80,
        value=40,
        step=1,
        help="Upper limit for forward P/E in buy-zone candidates.",
    )
    only_value = st.checkbox(
        "Only value signals (💚 / 🟡)",
        value=False,
        help="Filter to Deep value pullback and Value watch.",
    )
    buy_universe = st.radio(
        "Buy Zone Candidates",
        options=["Tech leaders only", "Nasdaq-100", "Both"],
        index=0,
    )


# -------------- TITLE + HEADER ------------------

st.title("Global Tech & Macro Dashboard")
st.caption(
    f"<p style='text-align:center;'>Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True,
)


# -------------- MACRO / SECTOR / GLOBAL STRIPS ------------------

US_MACRO_ETFS = [
    ("MARKET", "Market (QQQ / QQQ Futures)"),  # synthetic
    ("ARKK", "Disruptive Growth"),
    ("MEME", "Meme Beta"),
    ("SMH", "Semiconductors"),
    ("XLF", "Financials"),
    ("TLT", "Bonds"),
    ("UUP", "US Dollar"),
    ("GLD", "Gold"),
]

GLOBAL_INDICES = [
    ("^SSEC", "China (Shanghai)"),
    ("^KS11", "Korea (KOSPI)"),
    ("^N225", "Japan (Nikkei)"),
    ("^TWII", "Taiwan (TAIEX)"),
    ("^STOXX50E", "Europe (EuroStoxx50)"),
    ("^FTSE", "UK (FTSE 100)"),
    ("^GDAXI", "Germany (DAX)"),
    ("^FCHI", "France (CAC40)"),
]

# Status map for all real symbols
real_symbols = [t for t, _ in US_MACRO_ETFS if t != "MARKET"] + [t for t, _ in GLOBAL_INDICES]
status_map = {sym: get_ticker_status(sym) for sym in real_symbols}
status_map["MARKET"] = (active_mode, active_price, None, active_change_pct, active_arrow)

# China fallback
mode_ssec, price_ssec, chg_ssec, chg_pct_ssec, arr_ssec = status_map.get(
    "^SSEC", (None, None, None, None, None)
)
if price_ssec is None:
    fb = get_ticker_status("000001.SS")
    if fb[1] is not None:
        status_map["^SSEC"] = fb

market_state_map = {sym: get_market_state(sym) for sym in real_symbols}
market_state_map["MARKET"] = get_market_state(active_symbol)


st.markdown("### US Macro & Sector Pulse")


def render_card(label, ticker_display, status_tuple, market_state: str, show_state: bool):
    """
    Card renderer.

    - Colours label by % move.
    - If show_state is True and market_state == 'Closed' -> appends red '· Closed'.
    - Never prints 'Open' anywhere.
    """
    mode, price, _, chg_pct, arrow = status_tuple
    if price is None or chg_pct is None:
        html = (
            f"<div style='border:1px solid #1f2933; padding:0.5rem; "
            f"border-radius:0.75rem; background-color:#050505;'>"
            f"<div style='font-size:0.8rem; color:#9ca3af;'>{label}</div>"
            f"<div style='font-weight:600; color:#9ca3af;'>{ticker_display} data unavailable</div>"
            f"</div>"
        )
        return html

    if chg_pct > 0:
        txt_color = "#22c55e"
    elif chg_pct < 0:
        txt_color = "#ef4444"
    else:
        txt_color = "#e5e5e5"

    if show_state and market_state == "Closed":
        state_html = "<span style='color:#ef4444;'> · Closed</span>"
    else:
        state_html = ""

    html = (
        f"<div style='border:1px solid #1f2933; padding:0.5rem; "
        f"border-radius:0.75rem; background-color:#050505;'>"
        f"<div style='font-size:0.8rem; color:#9ca3af;'>{label}</div>"
        f"<div style='font-weight:600; color:{txt_color};'>"
        f"{ticker_display} {arrow} ({chg_pct:+.2f}%)"
        f"{state_html}"
        f"</div>"
        f"</div>"
    )
    return html


cards_per_row = 4
for i in range(0, len(US_MACRO_ETFS), cards_per_row):
    row = US_MACRO_ETFS[i: i + cards_per_row]
    cols = st.columns(len(row))
    for (ticker, label), col in zip(row, cols):
        with col:
            if ticker == "MARKET":
                display_ticker = active_label
                display_label = "Market (QQQ / QQQ Futures)"
                state_for_card = market_state_map["MARKET"]
                show_state = False
            else:
                display_ticker = ticker
                display_label = label
                state_for_card = market_state_map[ticker]
                show_state = True

            st.markdown(
                render_card(display_label, display_ticker, status_map[ticker], state_for_card, show_state),
                unsafe_allow_html=True,
            )

st.markdown("### Global Indices Pulse")

for i in range(0, len(GLOBAL_INDICES), cards_per_row):
    row = GLOBAL_INDICES[i: i + cards_per_row]
    cols = st.columns(len(row))
    for (ticker, label), col in zip(row, cols):
        with col:
            st.markdown(
                render_card(label, ticker, status_map[ticker], market_state_map[ticker], True),
                unsafe_allow_html=True,
            )


# -------------- TICKER UNIVERSES ------------------

TOP_TECH_TICKERS = [
    "MSFT",
    "AMZN",
    "GOOG",
    "NVDA",
    "META",
    "TSM",
    "AVGO",
    "ORCL",
    "CRM",
    "AMD",
    "NOW",
    "MU",
    "SNOW",
    "PLTR",
    "ANET",
    "CRWD",
    "PANW",
    "NET",
    "DDOG",
    "MDB",
    "MRVL",
    "IBM",
    "AMKR",
    "SMCI",
    "AXON",
    "ISRG",
]

NASDAQ100_TICKERS = [
    "ADBE",
    "AMD",
    "ABNB",
    "GOOGL",
    "GOOG",
    "AMZN",
    "AEP",
    "AMGN",
    "ADI",
    "AAPL",
    "AMAT",
    "APP",
    "ARM",
    "ASML",
    "AZN",
    "TEAM",
    "ADSK",
    "ADP",
    "AXON",
    "BKR",
    "BIIB",
    "BKNG",
    "AVGO",
    "CDNS",
    "CDW",
    "CHTR",
    "CTAS",
    "CSCO",
    "CCEP",
    "CTSH",
    "CMCSA",
    "CEG",
    "CPRT",
    "CSGP",
    "COST",
    "CRWD",
    "CSX",
    "DDOG",
    "DXCM",
    "FANG",
    "DASH",
    "EA",
    "EXC",
    "FAST",
    "FTNT",
    "GEHC",
    "GILD",
    "GFS",
    "HON",
    "IDXX",
    "INTC",
    "INTU",
    "ISRG",
    "KDP",
    "KLAC",
    "KHC",
    "LRCX",
    "LIN",
    "LULU",
    "MAR",
    "MRVL",
    "MELI",
    "META",
    "MCHP",
    "MU",
    "MSFT",
    "MSTR",
    "MDLZ",
    "MNST",
    "NFLX",
    "NVDA",
    "NXPI",
    "ORLY",
    "ODFL",
    "ON",
    "PCAR",
    "PLTR",
    "PANW",
    "PAYX",
    "PYPL",
    "PDD",
    "PEP",
    "QCOM",
    "REGN",
    "ROP",
    "ROST",
    "SHOP",
    "SOLS",
    "SBUX",
    "SNPS",
    "TMUS",
    "TTWO",
    "TSLA",
    "TXN",
    "TRI",
    "TTD",
    "VRSK",
    "VRTX",
    "WBD",
    "WDAY",
    "XEL",
    "ZS",
]

# Core watchlist for Focus Table
FOCUS_WATCHLIST = ["NVDA", "TSM", "AMD", "AVGO", "AMKR", "PLTR", "META"]


# -------------- HELPERS FOR TABLES ------------------


def get_value_momentum_signal(rsi, pct_from_high, pct_1m, fpe):
    if rsi is None or pct_from_high is None:
        return "❔ Check data"

    if rsi < 35 and pct_from_high <= -30 and (fpe is None or fpe <= 30):
        return "💚 Deep value pullback"

    if rsi < 50 and pct_from_high <= -15 and (fpe is None or fpe <= 35):
        return "🟡 Value watch"

    if 50 <= rsi <= 70 and (pct_1m is not None and pct_1m > 0):
        return "🔵 Momentum trend"

    if rsi > 70 or (pct_from_high >= -5 and (fpe is not None and fpe >= 45)):
        return "🔴 Hot / extended"

    return "⚪ Neutral"


def rsi_zone_text(rsi_val: float) -> str:
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
    if val is None:
        return ""
    try:
        num_str = str(val).split()[0]
        rsi = float(num_str)
    except Exception:
        return ""
    if rsi < 30:
        return "color: #22c55e; font-weight: 600;"
    if rsi < 50:
        return "color: #eab308; font-weight: 600;"
    if rsi < 70:
        return "color: #3b82f6; font-weight: 600;"
    return "color: #ef4444; font-weight: 600;"


# base colours for gradients
RED = (239, 68, 68)
BLACK = (0, 0, 0)
GREEN = (34, 197, 94)


def _blend(c_from, c_to, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(cf + (ct - cf) * t)) for cf, ct in zip(c_from, c_to))


def _rgb_css(c):
    return f"background-color: rgb({c[0]}, {c[1]}, {c[2]});"


def color_tripolar(v, vmin, vmax):
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
    if pd.isna(v) or vmin is None or vmax is None or vmin == vmax:
        return ""
    v = float(v)
    t = (v - vmin) / (vmax - vmin)
    col = _blend(RED, GREEN, t)
    return _rgb_css(col)


@st.cache_data(ttl=60)
def get_stock_summary(tickers):
    rows = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            _, price_rt, _, change_pct_rt, _ = get_ticker_status(ticker)

            hist = stock.history(period="1y")
            if hist.empty or "Close" not in hist.columns:
                continue

            close = hist["Close"].dropna()
            if len(close) < 10:
                continue

            last_close = float(close.iloc[-1])
            price = float(price_rt) if price_rt is not None else last_close

            pct_5d = (
                round((price - float(close.iloc[-6])) / float(close.iloc[-6]) * 100, 2)
                if len(close) >= 6
                else None
            )
            pct_1m = (
                round((price - float(close.iloc[-22])) / float(close.iloc[-22]) * 100, 2)
                if len(close) >= 22
                else None
            )

            high_52wk = float(close.max())
            pct_from_52wk = round((price - high_52wk) / high_52wk * 100, 2)

            rsi_series = RSIIndicator(close=close).rsi()
            rsi_val = float(round(rsi_series.iloc[-1], 2))

            try:
                info = stock.get_info()
            except Exception:
                info = {}

            pe = info.get("trailingPE", None)
            try:
                pe = float(pe)
            except Exception:
                pe = None

            market_cap = info.get("marketCap", None)
            try:
                market_cap = float(market_cap)
            except Exception:
                market_cap = None

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
                fpe=fpe,
            )

            rows.append(
                {
                    "Ticker": ticker,
                    "Price": price,
                    "% 1D": round(change_pct_rt, 2) if change_pct_rt is not None else None,
                    "% 5D": pct_5d,
                    "% 1M": pct_1m,
                    "% from 52w High": pct_from_52wk,
                    "RSI Zone": rsi_zone_str,
                    "Value Signal": value_signal,
                    "P/E": pe,
                    "Fwd P/E": fpe,
                    "Market Cap": market_cap,
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)


BASE_COLUMN_CONFIG = {
    col: st.column_config.Column(width="fit")
    for col in [
        "Price",        # still used in Buy-Zone section
        "Price & 1D",   # combined column for main tables
        "% 1D",
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
    cfg = {}
    for col in columns:
        if col in BASE_COLUMN_CONFIG:
            cfg[col] = BASE_COLUMN_CONFIG[col]
        else:
            cfg[col] = st.column_config.Column(width="fit")
    return cfg


# styling for separate Price/%1D (used in Buy-Zone candidates)
def price_style(row):
    val = row.get("% 1D", None)
    if pd.isna(val):
        return [""]
    if val > 0:
        return ["color: #22c55e; font-weight: 600;"]
    if val < 0:
        return ["color: #ef4444; font-weight: 600;"]
    return ["color: #e5e5e5; font-weight: 600;"]


def pct1d_style(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: #22c55e; font-weight: 600;"
    if val < 0:
        return "color: #ef4444; font-weight: 600;"
    return "color: #e5e5e5; font-weight: 600;"


# Style for combined "Price & 1D" column
def price_1d_style(val):
    """
    Style for 'Price & 1D' column:
    - green if 1D % > 0
    - red if 1D % < 0
    - grey otherwise
    """
    if val is None or val == "–":
        return ""
    try:
        inside = val.split("(")[1].split("%")[0]
        pct = float(inside)
    except Exception:
        return ""
    if pct > 0:
        return "color: #22c55e; font-weight: 600;"
    if pct < 0:
        return "color: #ef4444; font-weight: 600;"
    return "color: #e5e5e5; font-weight: 600;"


def format_price_1d(row):
    price = row["Price"]
    pct_1d = row["% 1D"]
    if pd.isna(price) and pd.isna(pct_1d):
        return "–"
    if pd.isna(pct_1d):
        return f"${price:,.2f}"
    return f"${price:,.2f} ({pct_1d:+.1f}%)"


# -------------- TABLE 1: TECH LEADERSHIP ------------------

df = pd.DataFrame()
df_ndx = pd.DataFrame()

st.markdown("---")
st.markdown("## Megacap & Core")

# Centered Focus Table checkbox
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    focus_only = st.checkbox(
        "Focus Table",
        value=False,
        help="Show only: NVDA, TSM, AMD, AVGO, AMKR, PLTR, META.",
    )

with st.spinner("📡 Fetching data for Tech leadership table..."):
    df = get_stock_summary(TOP_TECH_TICKERS)

if not df.empty:
    df = df.set_index("Ticker")

    # Apply Focus filter if enabled
    if focus_only:
        df = df[df.index.isin(FOCUS_WATCHLIST)]

    if "Market Cap" in df.columns:
        df_sorted = df.sort_values("Market Cap", ascending=False)
    else:
        df_sorted = df.copy()

    df_display = df_sorted.drop(columns=["Market Cap"], errors="ignore")

    # combined column
    df_display["Price & 1D"] = df_display.apply(format_price_1d, axis=1)
    df_display = df_display.drop(columns=["Price", "% 1D"])

    # REORDER: move Price & 1D next to Ticker (first data column)
    desired_order = [
        "Price & 1D",
        "% 5D",
        "% 1M",
        "% from 52w High",
        "RSI Zone",
        "Value Signal",
        "P/E",
        "Fwd P/E",
    ]
    existing_cols = [c for c in desired_order if c in df_display.columns]
    df_display = df_display[existing_cols]

    format_dict = {
        "Price & 1D": "{}",
        "% 5D": "{:.1f}%",
        "% 1M": "{:.1f}%",
        "% from 52w High": "{:.1f}%",
        "P/E": "{:.1f}",
        "Fwd P/E": "{:.1f}",
    }

    styled = df_display.style.format(format_dict, na_rep="–")

    pct_cols = ["% 5D", "% 1M"]
    dist_col = "% from 52w High"

    for col in pct_cols:
        if df_display[col].notna().any():
            vmin = df_display[col].min()
            vmax = df_display[col].max()
            styled = styled.apply(
                lambda s, vmin=vmin, vmax=vmax: [color_tripolar(v, vmin, vmax) for v in s],
                subset=[col],
                axis=0,
            )

    if df_display[dist_col].notna().any():
        vmin = df_display[dist_col].min()
        vmax = 0.0
        styled = styled.apply(
            lambda s, vmin=vmin, vmax=vmax: [color_bipolar(v, vmin, vmax) for v in s],
            subset=[dist_col],
            axis=0,
        )

    styled = styled.set_table_styles(
        [
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ],
        overwrite=False,
    )

    styled = styled.applymap(rsi_zone_style, subset=["RSI Zone"])
    styled = styled.applymap(price_1d_style, subset=["Price & 1D"])

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
    "<h2>NASDAQ 100 DEEP DRAWDOWN RADAR</h2>",
    unsafe_allow_html=True,
)

with st.spinner("📡 Fetching Nasdaq-100 data..."):
    df_ndx = get_stock_summary(NASDAQ100_TICKERS)

if not df_ndx.empty:
    df_ndx = df_ndx.set_index("Ticker")

    # Apply Focus filter here too
    if focus_only:
        df_ndx = df_ndx[df_ndx.index.isin(FOCUS_WATCHLIST)]

    df_ndx = df_ndx.sort_values("% from 52w High")
    df_ndx_display = df_ndx.drop(columns=["Market Cap"], errors="ignore")

    # combined column for Nasdaq table
    df_ndx_display["Price & 1D"] = df_ndx_display.apply(format_price_1d, axis=1)
    df_ndx_display = df_ndx_display.drop(columns=["Price", "% 1D"])

    # REORDER: Price & 1D first
    desired_order_ndx = [
        "Price & 1D",
        "% 5D",
        "% 1M",
        "% from 52w High",
        "RSI Zone",
        "Value Signal",
        "P/E",
        "Fwd P/E",
    ]
    existing_ndx = [c for c in desired_order_ndx if c in df_ndx_display.columns]
    df_ndx_display = df_ndx_display[existing_ndx]

    ndx_format_dict = {
        "Price & 1D": "{}",
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
                lambda s, vmin=vmin, vmax=vmax: [color_tripolar(v, vmin, vmax) for v in s],
                subset=[col],
                axis=0,
            )

    if df_ndx_display[ndx_dist_col].notna().any():
        vmin = df_ndx_display[ndx_dist_col].min()
        vmax = 0.0
        styled_ndx = styled_ndx.apply(
            lambda s, vmin=vmin, vmax=vmax: [color_bipolar(v, vmin, vmax) for v in s],
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
    styled_ndx = styled_ndx.applymap(price_1d_style, subset=["Price & 1D"])

    ndx_column_config = build_column_config(df_ndx_display.columns)

    st.dataframe(
        styled_ndx,
        use_container_width=True,
        height=600,
        column_config=ndx_column_config,
    )
else:
    st.write("No Nasdaq-100 data loaded.")


# -------------- BUY-ZONE CANDIDATES ------------------

st.markdown("---")
st.markdown("## Buy-Zone Candidates (Screened by Your Rules)")


def build_buy_candidates(df_tech, df_nasdaq):
    sources = []
    if buy_universe in ("Tech leaders only", "Both") and df_tech is not None and not df_tech.empty:
        sources.append(df_tech.copy())
    if buy_universe in ("Nasdaq-100", "Both") and df_nasdaq is not None and not df_nasdaq.empty:
        sources.append(df_nasdaq.copy())

    if not sources:
        return pd.DataFrame()

    base = pd.concat(sources, axis=0)
    base = base[~base.index.duplicated(keep="first")]

    # Apply Focus filter here too
    if focus_only:
        base = base[base.index.isin(FOCUS_WATCHLIST)]

    rsi_numeric = base["RSI Zone"].str.extract(r"([\d.]+)").astype(float)[0]
    base["RSI_numeric"] = rsi_numeric

    mask = pd.Series(True, index=base.index)

    mask &= base["% from 52w High"] <= min_dd
    mask &= base["RSI_numeric"] < 55
    mask &= base["Fwd P/E"].notna()
    mask &= base["Fwd P/E"] <= float(max_fpe)

    if only_value:
        mask &= base["Value Signal"].str.contains(
            "Deep value pullback|Value watch", na=False
        )

    candidates = base.loc[mask].copy()
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values("% from 52w High")
    return candidates


if df is not None and df_ndx is not None:
    candidates = build_buy_candidates(df, df_ndx)
else:
    candidates = pd.DataFrame()

if not candidates.empty:
    show_cols = ["Price", "% 1D", "% from 52w High", "RSI Zone", "Fwd P/E", "Value Signal"]
    candidates_display = candidates[show_cols]

    cand_format = {
        "Price": "${:,.2f}",
        "% 1D": "{:.1f}%",
        "% from 52w High": "{:.1f}%",
        "Fwd P/E": "{:.1f}",
    }

    cand_styled = candidates_display.style.format(cand_format, na_rep="–")
    cand_styled = cand_styled.set_table_styles(
        [
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ],
        overwrite=False,
    )
    cand_styled = cand_styled.applymap(rsi_zone_style, subset=["RSI Zone"])
    cand_styled = cand_styled.apply(lambda row: price_style(row), subset=["Price"], axis=1)
    cand_styled = cand_styled.applymap(pct1d_style, subset=["% 1D"])

    st.dataframe(
        cand_styled,
        use_container_width=True,
        height=400,
    )
else:
    st.write(
        "No tickers currently match your buy-zone criteria. "
        "Loosen filters in the sidebar to widen the search."
    )


# -------------- HOW TO READ THE SIGNALS ------------------

st.markdown("---")
st.markdown(
    """
**Value Signal (combined value + momentum)**  
- 💚 **Deep value pullback** – Big drawdown vs 52-week high, low or reasonable forward P/E, weak RSI.  
- 🟡 **Value watch** – Decent pullback, valuation reasonable but not screaming.  
- 🔵 **Momentum trend** – Positive 1-month performance with RSI in 50–70 zone.  
- 🔴 **Hot / extended** – Near highs and/or expensive forward P/E, or overbought RSI.  
- ⚪ **Neutral** – No strong edge from value or momentum.
"""
)
