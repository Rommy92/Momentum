import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
from ta.momentum import RSIIndicator
from pandas import IndexSlice  # for Styler.map subset

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


@st.cache_data(ttl=30)
def get_ticker_status(symbol: str):
    """
    Realtime-like price status with full extended-hours support.

    Priority 1: Use marketState + pre/post/regular fields from Yahoo get_info().
    Priority 2: Fall back to intraday 1m data (prepost=True).

    Returns (mode, price, change, change_pct, arrow).
    """
    # -------- Path 1: Use get_info() with marketState --------
    try:
        t = yf.Ticker(symbol)
        info = t.get_info() or {}
        state = str(info.get("marketState", "")).upper()

        price = change = change_pct = None

        if state == "PRE":
            price = info.get("preMarketPrice") or info.get("regularMarketPrice")
            change = info.get("preMarketChange")
            change_pct = info.get("preMarketChangePercent")

        elif state == "POST":
            price = info.get("postMarketPrice") or info.get("regularMarketPrice")
            change = info.get("postMarketChange")
            change_pct = info.get("postMarketChangePercent")

        else:  # REGULAR, CLOSED, or anything else
            price = info.get("regularMarketPrice")
            change = info.get("regularMarketChange")
            change_pct = info.get("regularMarketChangePercent")

        if price is not None and change_pct is not None:
            price = float(price)
            change_pct = float(change_pct)

            if change is None:
                try:
                    prev_close = price / (1 + change_pct / 100.0)
                    change = price - prev_close
                except Exception:
                    change = 0.0
            else:
                change = float(change)

            if change > 0:
                return "green", price, change, change_pct, "▲"
            if change < 0:
                return "red", price, change, change_pct, "▼"
            return "neutral", price, change, change_pct, "▶"

    except Exception:
        # fall through to intraday path
        pass

    # -------- Path 2: Use 1m intraday as fallback --------
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
            return "green", price, change, change_pct, "▲"
        if change < 0:
            return "red", price, change, change_pct, "▼"
        return "neutral", price, change, change_pct, "▶"

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

# -------------- THEME SETUP ------------------

THEMES = ["Terminal", "Modern", "Original"]

if "theme" not in st.session_state:
    # Default theme: Terminal
    st.session_state["theme"] = "Terminal"


def set_theme(name: str):
    st.session_state["theme"] = name


current_theme = st.session_state["theme"]


def get_theme_css(theme: str, accent_color: str) -> str:
    # Modern: brutalist Palantir energy
    if theme == "Modern":
        return f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 0% 0%, #111827 0, #020617 35%, #000000 100%),
                repeating-linear-gradient(
                    135deg,
                    rgba(30,64,175,0.16) 0px,
                    rgba(30,64,175,0.16) 1px,
                    transparent 1px,
                    transparent 10px
                ) !important;
            color: #e5e7eb !important;
        }}
        [data-testid="stSidebar"] {{
            background: #020617 !important;
            color: #e5e7eb !important;
            border-right: 1px solid rgba(148,163,184,0.5) !important;
        }}
        html, body, [class*="css"] {{
            color: #e5e7eb !important;
            background-color: transparent !important;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif !important;
        }}
        h1 {{
            color: #f9fafb !important;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.95rem;
            border-bottom: 2px solid #ef4444;
            padding-bottom: 0.35rem;
            margin-bottom: 0.2rem;
        }}
        h2 {{
            color: #e5e7eb !important;
            text-align: left;
            font-size: 0.8rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }}
        h3, h4 {{
            color: #f97316 !important;
            text-align: left;
            font-size: 0.75rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }}
        .block-container {{
            padding-top: 2.2rem !important;
            padding-bottom: 1.5rem !important;
            max-width: 1350px !important;
        }}
        .modern-panel {{
            background: #020617;
            border-radius: 1.2rem;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow:
                0 18px 40px rgba(0,0,0,0.95),
                0 0 0 1px rgba(15,23,42,0.85);
            padding: 0.9rem 1.1rem;
            position: relative;
            overflow: hidden;
        }}
        .modern-panel::before {{
            content: "";
            position: absolute;
            inset: -40%;
            background:
                radial-gradient(circle at 0 0, rgba(239,68,68,0.14), transparent 55%),
                radial-gradient(circle at 100% 100%, rgba(59,130,246,0.15), transparent 60%);
            opacity: 0.7;
            pointer-events: none;
        }}
        .modern-panel-inner {{
            position: relative;
            z-index: 1;
        }}
        .hero-kpi {{
            border-radius: 0.9rem;
            padding: 0.7rem 0.9rem;
            background: linear-gradient(135deg, #020617, #020617);
            border: 1px solid rgba(249,115,22,0.6);
            box-shadow:
                0 12px 40px rgba(0,0,0,0.95),
                0 0 0 1px rgba(15,23,42,0.9);
        }}
        .hero-label {{
            font-size: 0.72rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #9ca3af;
        }}
        .hero-value {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #f9fafb;
        }}
        .hero-sub {{
            font-size: 0.72rem;
            color: #6b7280;
        }}
        [data-testid="stDataFrame"] div[role="grid"] {{
            background: #020617 !important;
            color: #e5e7eb !important;
            border-radius: 0.9rem !important;
            border: 1px solid rgba(75,85,99,0.9) !important;
        }}
        [data-testid="stDataFrame"] div[role="columnheader"] {{
            background: #020617 !important;
            color: #e5e7eb !important;
            border-bottom: 1px solid rgba(55,65,81,0.95) !important;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.13em;
        }}
        [data-testid="stDataFrame"] div[role="cell"] {{
            border-bottom: 1px solid #0f172a !important;
        }}
        </style>
        """

    # Terminal: dark terminal, monospaced, yellow headers
    if theme == "Terminal":
        return """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #000000 !important;
            color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] {
            background-color: #020617 !important;
            color: #e5e7eb !important;
            border-right: 1px solid #1f2937 !important;
        }
        html, body, [class*="css"] {
            color: #e5e7eb !important;
            background-color: #000000 !important;
            font-family: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
        }
        h1, h2 {
            color: #facc15 !important;
            text-shadow: none !important;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.16em;
        }
        h3, h4 {
            color: #f97316 !important;
            text-align: left;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.18em;
        }
        .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 1.25rem !important;
            max-width: 100% !important;
        }
        [data-testid="stDataFrame"] div[role="grid"] {
            background-color: #020617 !important;
            color: #e5e7eb !important;
            border-radius: 0 !important;
            border: 1px solid #1f2937 !important;
        }
        [data-testid="stDataFrame"] div[role="columnheader"] {
            background-color: #111827 !important;
            color: #facc15 !important;
            border-bottom: 1px solid #4b5563 !important;
            text-transform: uppercase;
            font-size: 0.75rem;
        }
        [data-testid="stDataFrame"] div[role="cell"] {
            border-bottom: 1px solid #111827 !important;
        }
        </style>
        """

    # Original (Cyberpunk): neon, heatmaps
    return f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: #000000 !important;
        color: #eeeeee !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: #000000 !important;
        color: #eeeeee !important;
        border-right: 1px solid {accent_color}33 !important;
    }}
    html, body, [class*="css"] {{
        color: #eeeeee !important;
        background-color: #000000 !important;
    }}
    h1, h2 {{
        color: {accent_color} !important;
        text-shadow: 0 0 4px {accent_color}, 0 0 10px {accent_color};
        animation: neonPulse 3s ease-in-out infinite;
        text-align: center;
    }}
    h3, h4 {{
        color: {accent_color} !important;
        text-align: center;
    }}
    @keyframes neonPulse {{
        0% {{
            text-shadow: 0 0 4px {accent_color}, 0 0 8px {accent_color};
        }}
        50% {{
            text-shadow: 0 0 12px {accent_color}, 0 0 22px {accent_color};
        }}
        100% {{
            text-shadow: 0 0 4px {accent_color}, 0 0 8px {accent_color};
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
        color: {accent_color} !important;
        border-bottom: 1px solid {accent_color}77 !important;
    }}
    [data-testid="stDataFrame"] div[role="cell"] {{
        border-bottom: 1px solid #222222 !important;
    }}
    </style>
    """


theme_css = get_theme_css(current_theme, accent)
st.markdown(theme_css, unsafe_allow_html=True)

# -------------- SIDEBAR FILTERS ------------------

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

# -------------- TITLE + THEME BUTTONS ------------------

st.title("Global Tech & Macro Dashboard")
st.caption(
    f"<p style='text-align:center;'>Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True,
)

# Push buttons a bit down so they are visible in Streamlit
st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

# THEME BUTTONS: no tick, just highlight bar under active theme
theme_cols = st.columns(len(THEMES))
for col, name in zip(theme_cols, THEMES):
    with col:
        is_current = st.session_state["theme"] == name
        if st.button(name, key=f"theme_{name}", use_container_width=True):
            set_theme(name)
            is_current = True
        bar_color = "#facc15" if is_current else "transparent"
        st.markdown(
            f"<div style='height:3px; border-radius:999px; margin-top:4px; background:{bar_color};'></div>",
            unsafe_allow_html=True,
        )

current_theme = st.session_state["theme"]

# -------------- UNIVERSES ------------------

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
    ("000001.SS", "China (Shanghai)"),
    ("^KS11", "Korea (KOSPI)"),
    ("^N225", "Japan (Nikkei)"),
    ("^TWII", "Taiwan (TAIEX)"),
    ("^STOXX50E", "Europe (EuroStoxx50)"),
    ("^FTSE", "UK (FTSE 100)"),
    ("^GDAXI", "Germany (DAX)"),
    ("^FCHI", "France (CAC40)"),
]

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
    "SYM",
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

FOCUS_TICKERS = ["NVDA", "TSM", "AMD", "AVGO", "AMKR", "PLTR", "META"]

# Trade universe for swing ideas
TRADE_UNIVERSE = [
    "NVDA", "AMD", "AVGO", "TSM", "MSFT", "META", "GOOG", "AMZN",
    "AAPL", "CRM", "NFLX",
    "SMCI", "MRVL", "PANW", "DDOG", "NOW",
]

# -------------- STATUS MAPS ------------------

real_symbols = [t for t, _ in US_MACRO_ETFS if t != "MARKET"] + [t for t, _ in GLOBAL_INDICES]
status_map = {sym: get_ticker_status(sym) for sym in real_symbols}
status_map["MARKET"] = (active_mode, active_price, None, active_change_pct, active_arrow)

market_state_map = {sym: get_market_state(sym) for sym in real_symbols}
market_state_map["MARKET"] = get_market_state(active_symbol)

# -------------- TABLE HELPERS ------------------


def get_value_momentum_signal(rsi, pct_from_high, pct_1m, fpe):
    if pct_from_high is None:
        return "❔ Check data"

    if (
        pct_from_high <= -30
        and (fpe is None or fpe <= 30)
        and (pct_1m is None or pct_1m <= 10)
    ):
        return "💚 Deep value pullback"

    if pct_from_high <= -15 and (fpe is None or fpe <= 35):
        return "🟡 Value watch"

    if pct_1m is not None and pct_1m > 0 and pct_from_high >= -25:
        return "🔵 Momentum trend"

    if (
        pct_from_high >= -5
        and fpe is not None
        and fpe >= 45
        and (pct_1m is None or pct_1m >= 0)
    ):
        return "🔴 Hot / extended"

    return "⚪ Neutral"


def compute_vm_score(rsi, pct_from_high, pct_1m, fpe):
    if fpe is None:
        val_points = 1
    else:
        if fpe <= 20:
            val_points = 3
        elif fpe <= 30:
            val_points = 2
        elif fpe <= 40:
            val_points = 1
        else:
            val_points = 0

    dd_points = 0
    if pct_from_high is not None:
        if pct_from_high <= -40:
            dd_points = 3
        elif pct_from_high <= -30:
            dd_points = 2
        elif pct_from_high <= -20:
            dd_points = 1

    mom_points = 0
    if pct_1m is not None:
        if pct_1m > 0:
            mom_points += 1
        if pct_1m > 10:
            mom_points += 1
    mom_points = min(mom_points, 2)

    score = val_points + dd_points + mom_points

    hot = False
    if (
        pct_from_high is not None
        and pct_from_high >= -5
        and fpe is not None
        and fpe >= 35
    ):
        hot = True
    if (
        pct_1m is not None
        and pct_1m > 15
        and fpe is not None
        and fpe >= 40
    ):
        hot = True

    if hot:
        score -= 3

    score = max(0, min(8, score))
    return score


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


RED = (239, 68, 68)
BLACK = (0, 0, 0)
GREEN = (34, 197, 94)


def _blend(c_from, c_to, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(cf + (ct - cf) * t)) for cf, ct in zip(c_from, c_to))


def _rgb_css(c):
    return f"background-color: rgb({c[0]}, {c[1]}, {c[2]});"


def color_tripolar(v, vmin, vmax):
    theme = st.session_state.get("theme", "Original")
    # For Modern, keep tables visually cleaner (no gradient heatmap)
    if theme == "Modern":
        return ""
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
    theme = st.session_state.get("theme", "Original")
    if theme == "Modern":
        return ""
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
            if len(close) < 30:
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

            # EMAs
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()

            prev_close = float(close.iloc[-2])
            prev_ema9 = float(ema9.iloc[-2])
            last_ema9 = float(ema9.iloc[-1])
            prev_ema20 = float(ema20.iloc[-2])
            last_ema20 = float(ema20.iloc[-1])

            ema9_reclaim = (last_close > last_ema9) and (prev_close <= prev_ema9)
            ema20_bounce = (last_close > last_ema20) and (prev_close <= prev_ema20)

            recent_high_20 = float(close.iloc[-20:].max())
            breakout_retest = (last_close >= recent_high_20) and (
                (last_close - recent_high_20) / recent_high_20 <= 0.05
            )

            rsi_series = RSIIndicator(close=close).rsi()
            rsi_val = float(round(rsi_series.iloc[-1], 2))

            rsi_reset = 35 <= rsi_val <= 55

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

            vm_score_raw = compute_vm_score(
                rsi=rsi_val,
                pct_from_high=pct_from_52wk,
                pct_1m=pct_1m,
                fpe=fpe,
            )

            vm_score_display = f"{vm_score_raw} – {value_signal}"

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
                    "VM Score Raw": vm_score_raw,
                    "VM Score": vm_score_display,
                    "P/E": pe,
                    "Fwd P/E": fpe,
                    "Market Cap": market_cap,
                    "EMA9": last_ema9,
                    "9 EMA Reclaim": ema9_reclaim,
                    "20 EMA Bounce": ema20_bounce,
                    "Breakout Retest": breakout_retest,
                    "RSI Reset": rsi_reset,
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)


BASE_COLUMN_CONFIG = {
    col: st.column_config.Column(width="fit")
    for col in [
        "Price",
        "Price & 1D",
        "% 1D",
        "% 5D",
        "% 1M",
        "% from 52w High",
        "RSI Zone",
        "VM Score",
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


def price_style(row):
    val = row.get("% 1D", None)
    if pd.isna(val):
        return [""]
    if val > 0:
        return ["color: #22c55e; font-weight: 600;"]
    if val < 0:
        return ["color: #ef4444; font-weight: 600;"]
    return ["color: #6b7280; font-weight: 600;"]


def pct1d_style(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: #22c55e; font-weight: 600;"
    if val < 0:
        return "color: #ef4444; font-weight: 600;"
    return "color: #6b7280; font-weight: 600;"


def price_1d_style(val):
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


def vm_score_style(val):
    if val is None or val == "–":
        return ""
    txt = str(val)
    score = None
    try:
        first_part = txt.split("–")[0].strip()
        score = int(first_part)
    except Exception:
        score = None

    if "💚" in txt:
        return "color: #22c55e; font-weight: 800;"
    if "🟡" in txt:
        return "color: #eab308; font-weight: 700;"
    if "🔵" in txt:
        return "color: #3b82f6; font-weight: 700;"
    if "🔴" in txt:
        return "color: #ef4444; font-weight: 700;"

    if score is not None:
        if score >= 6:
            return "color: #22c55e; font-weight: 800;"
        if score >= 4:
            return "color: #eab308; font-weight: 700;"
        if score >= 2:
            return "color: #3b82f6; font-weight: 600;"
        return "color: #9ca3af; font-weight: 500;"

    return ""


def setup_strength_style(val):
    if val == "High":
        return "color: #22c55e; font-weight: 700;"
    if val == "Medium":
        return "color: #eab308; font-weight: 600;"
    if val == "Low":
        return "color: #9ca3af; font-weight: 500;"
    return ""


def format_ema9_display(ema_val, reclaim_flag):
    if pd.isna(ema_val):
        return "–"
    try:
        v = float(ema_val)
    except Exception:
        return "–"
    val_str = f"{v:.2f}"
    if bool(reclaim_flag):
        return f"{val_str} (reclaim)"
    return val_str


# -------------- MACRO RENDERING ------------------


def render_card(label, ticker_display, status_tuple, market_state: str, show_state: bool, theme: str):
    mode, price, _, chg_pct, arrow = status_tuple

    if theme == "Modern":
        card_bg = "#020617"
        border = "rgba(75,85,99,0.95)"
        label_color = "#9ca3af"
    elif theme == "Terminal":
        card_bg = "#020617"
        border = "#1f2937"
        label_color = "#9ca3af"
    else:
        card_bg = "#050505"
        border = "#1f2933"
        label_color = "#9ca3af"

    if price is None or chg_pct is None:
        html = (
            f"<div style='border:1px solid {border}; padding:0.5rem; "
            f"border-radius:{'0.25rem' if theme=='Terminal' else '0.75rem'}; "
            f"background-color:{card_bg};'>"
            f"<div style='font-size:0.8rem; color:{label_color};'>{label}</div>"
            f"<div style='font-weight:600; color:{label_color};'>{ticker_display} data unavailable</div>"
            f"</div>"
        )
        return html

    if chg_pct > 0:
        txt_color = "#22c55e"
    elif chg_pct < 0:
        txt_color = "#ef4444"
    else:
        txt_color = "#e5e5e5" if theme != "Modern" else "#cbd5f5"

    if show_state and market_state == "Closed":
        state_html = "<span style='color:#ef4444;'> · Closed</span>"
    else:
        state_html = ""

    html = (
        f"<div style='border:1px solid {border}; padding:0.5rem; "
        f"border-radius:{'0.25rem' if theme=='Terminal' else '0.75rem'}; "
        f"background-color:{card_bg};'>"
        f"<div style='font-size:0.8rem; color:{label_color};'>{label}</div>"
        f"<div style='font-weight:600; color:{txt_color};'>"
        f"{ticker_display} {arrow} ({chg_pct:+.2f}%)"
        f"{state_html}"
        f"</div>"
        f"</div>"
    )
    return html


def render_macro_section(theme: str):
    st.markdown("### US Macro & Sector Pulse")
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
                    render_card(display_label, display_ticker, status_map[ticker], state_for_card, show_state, theme),
                    unsafe_allow_html=True,
                )

    st.markdown("### Global Indices Pulse")
    cards_per_row = 4
    for i in range(0, len(GLOBAL_INDICES), cards_per_row):
        row = GLOBAL_INDICES[i: i + cards_per_row]
        cols = st.columns(len(row))
        for (ticker, label), col in zip(row, cols):
            with col:
                st.markdown(
                    render_card(label, ticker, status_map[ticker], market_state_map[ticker], True, theme),
                    unsafe_allow_html=True,
                )


# -------------- TABLE RENDERERS ------------------


def render_megacap_table(df: pd.DataFrame, accent: str, focus_mode: bool, height: int = 600):
    if df.empty:
        st.write("No data loaded.")
        return

    df = df.set_index("Ticker")

    if focus_mode:
        df = df.loc[df.index.intersection(FOCUS_TICKERS)]

    if "Market Cap" in df.columns:
        df_sorted = df.sort_values("Market Cap", ascending=False)
    else:
        df_sorted = df.copy()

    df_display = df_sorted.drop(columns=["Market Cap", "VM Score Raw"], errors="ignore")

    df_display["Price & 1D"] = df_display.apply(format_price_1d, axis=1)
    df_display = df_display.drop(columns=["Price", "% 1D"], errors="ignore")

    desired_order = [
        "Price & 1D",
        "% 5D",
        "% 1M",
        "% from 52w High",
        "RSI Zone",
        "VM Score",
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

    styled = styled.map(rsi_zone_style, subset=IndexSlice[:, ["RSI Zone"]])
    styled = styled.map(price_1d_style, subset=IndexSlice[:, ["Price & 1D"]])
    styled = styled.map(vm_score_style, subset=IndexSlice[:, ["VM Score"]])

    styled = styled.set_table_styles(
        [
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ],
        overwrite=False,
    )

    column_config = build_column_config(df_display.columns)

    st.dataframe(
        styled,
        width="stretch",
        height=height,
        column_config=column_config,
    )


def render_nasdaq_table(df_ndx: pd.DataFrame, focus_mode: bool, height: int = 600):
    if df_ndx.empty:
        st.write("No Nasdaq-100 data loaded.")
        return

    df_ndx = df_ndx.set_index("Ticker")

    if focus_mode:
        df_ndx = df_ndx.loc[df_ndx.index.intersection(FOCUS_TICKERS)]

    df_ndx = df_ndx.sort_values("% from 52w High")
    df_ndx_display = df_ndx.drop(columns=["Market Cap", "VM Score Raw"], errors="ignore")

    df_ndx_display["Price & 1D"] = df_ndx_display.apply(format_price_1d, axis=1)
    df_ndx_display = df_ndx_display.drop(columns=["Price", "% 1D"], errors="ignore")

    desired_order_ndx = [
        "Price & 1D",
        "% 5D",
        "% 1M",
        "% from 52w High",
        "RSI Zone",
        "VM Score",
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

    styled_ndx = styled_ndx.map(rsi_zone_style, subset=IndexSlice[:, ["RSI Zone"]])
    styled_ndx = styled_ndx.map(price_1d_style, subset=IndexSlice[:, ["Price & 1D"]])
    styled_ndx = styled_ndx.map(vm_score_style, subset=IndexSlice[:, ["VM Score"]])

    styled_ndx = styled_ndx.set_table_styles(
        [
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ],
        overwrite=False,
    )

    ndx_column_config = build_column_config(df_ndx_display.columns)

    st.dataframe(
        styled_ndx,
        width="stretch",
        height=height,
        column_config=ndx_column_config,
    )


def build_buy_candidates(df_tech, df_nasdaq, focus_mode: bool):
    sources = []
    if buy_universe in ("Tech leaders only", "Both") and df_tech is not None and not df_tech.empty:
        sources.append(df_tech.copy())
    if buy_universe in ("Nasdaq-100", "Both") and df_nasdaq is not None and not df_nasdaq.empty:
        sources.append(df_nasdaq.copy())

    if not sources:
        return pd.DataFrame()

    base = pd.concat(sources, axis=0)
    base = base[~base.index.duplicated(keep="first")]

    if focus_mode:
        base = base.loc[base.index.intersection(FOCUS_TICKERS)]

    mask = pd.Series(True, index=base.index)

    mask &= base["% from 52w High"] <= min_dd

    mask &= base["Fwd P/E"].notna()
    mask &= base["Fwd P/E"] <= float(max_fpe)

    if only_value:
        mask &= base["Value Signal"].str.contains(
            "Deep value pullback|Value watch", na=False
        )

    candidates = base.loc[mask].copy()
    if candidates.empty:
        return candidates

    if "VM Score Raw" in candidates.columns:
        candidates = candidates.sort_values(["VM Score Raw", "% from 52w High"], ascending=[False, True])
    else:
        candidates = candidates.sort_values("% from 52w High")

    return candidates


def render_buy_zone(df, df_ndx, focus_mode: bool, height: int = 400):
    if df is None or df_ndx is None:
        st.write("No data for candidates.")
        return

    candidates = build_buy_candidates(df, df_ndx, focus_mode)
    if candidates.empty:
        st.write(
            "No tickers currently match your buy-zone criteria. "
            "Loosen filters in the sidebar to widen the search."
        )
        return

    candidates["Price & 1D"] = candidates.apply(format_price_1d, axis=1)

    show_cols = [
        "Price",
        "% 1D",
        "% from 52w High",
        "RSI Zone",
        "Fwd P/E",
        "VM Score",
    ]
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
    cand_styled = cand_styled.map(rsi_zone_style, subset=IndexSlice[:, ["RSI Zone"]])
    cand_styled = cand_styled.apply(lambda row: price_style(row), subset=["Price"], axis=1)
    cand_styled = cand_styled.map(pct1d_style, subset=IndexSlice[:, ["% 1D"]])
    cand_styled = cand_styled.map(vm_score_style, subset=IndexSlice[:, ["VM Score"]])

    st.dataframe(
        cand_styled,
        width="stretch",
        height=height,
    )


def render_how_to():
    st.markdown("---")
    st.markdown(
        """
**VM Score (0–8) – combined value + drawdown + 1M price action, with hot penalty**  
- Higher score = better blend of: cheap / beaten-up / stabilising / not overheated.  

**Value Signal (inside VM column)**  
- 💚 Deep value pullback – Big drawdown vs 52-week high, low or reasonable forward P/E, and not already ripping.  
- 🟡 Value watch – Decent pullback, valuation reasonable but not screaming.  
- 🔵 Momentum trend – Positive 1-month performance with moderate drawdown.  
- 🔴 Hot / extended – Near highs and/or expensive forward P/E with recent strength.  
- ⚪ Neutral – No strong edge from value or price action.

**RSI Zone** is informational only – context, not used directly in the VM Score or filters.
"""
    )


# -------------- TRADE IDEAS ------------------


def render_trade_ideas(df: pd.DataFrame, height: int = 400):
    if df is None or df.empty:
        st.write("No data for trade ideas.")
        return

    df = df[df["Ticker"].isin(TRADE_UNIVERSE)].copy()
    if df.empty:
        st.write("No tickers from the trade universe are in the dataset.")
        return

    # Primary trigger: EMA9 reclaim
    df = df[df["9 EMA Reclaim"] == True].copy()
    if df.empty:
        st.write("No tickers currently reclaiming the 9-day EMA based on your universe.")
        return

    def setup_strength(row):
        flags = 0
        if row.get("9 EMA Reclaim"):
            flags += 1
        if row.get("20 EMA Bounce"):
            flags += 1
        if row.get("Breakout Retest"):
            flags += 1
        rsi_reset = bool(row.get("RSI Reset"))

        if flags >= 2 and rsi_reset:
            return "High"
        if flags >= 1 and rsi_reset:
            return "Medium"
        return "Low"

    df["Setup Strength"] = df.apply(setup_strength, axis=1)

    df_trades = df.set_index("Ticker")

    df_trades["Price & 1D"] = df_trades.apply(format_price_1d, axis=1)
    df_trades["EMA9"] = df_trades.apply(
        lambda r: format_ema9_display(r.get("EMA9", None), r.get("9 EMA Reclaim", False)),
        axis=1,
    )

    df_trades["20EMA Bounce"] = df_trades["20 EMA Bounce"].apply(
        lambda x: "Yes" if bool(x) else "–"
    )
    df_trades["Breakout Retest"] = df_trades["Breakout Retest"].apply(
        lambda x: "Yes" if bool(x) else "–"
    )
    df_trades["RSI Reset"] = df_trades["RSI Reset"].apply(
        lambda x: "Yes" if bool(x) else "–"
    )

    cols_order = [
        "Price & 1D",
        "EMA9",
        "Setup Strength",
        "% 5D",
        "% 1M",
        "% from 52w High",
        "20EMA Bounce",
        "Breakout Retest",
        "RSI Reset",
    ]
    existing_cols = [c for c in cols_order if c in df_trades.columns]
    df_trades_display = df_trades[existing_cols]

    format_dict = {
        "% 5D": "{:.1f}%",
        "% 1M": "{:.1f}%",
        "% from 52w High": "{:.1f}%",
    }

    styled = df_trades_display.style.format(format_dict, na_rep="–")

    # Heatmaps for returns and drawdown
    for col in ["% 5D", "% 1M"]:
        if col in df_trades_display.columns and df_trades_display[col].notna().any():
            vmin = df_trades_display[col].min()
            vmax = df_trades_display[col].max()
            styled = styled.apply(
                lambda s, vmin=vmin, vmax=vmax: [color_tripolar(v, vmin, vmax) for v in s],
                subset=[col],
                axis=0,
            )

    if "% from 52w High" in df_trades_display.columns and df_trades_display["% from 52w High"].notna().any():
        vmin = df_trades_display["% from 52w High"].min()
        vmax = 0.0
        styled = styled.apply(
            lambda s, vmin=vmin, vmax=vmax: [color_bipolar(v, vmin, vmax) for v in s],
            subset=["% from 52w High"],
            axis=0,
        )

    # Styles
    if "Price & 1D" in df_trades_display.columns:
        styled = styled.map(price_1d_style, subset=IndexSlice[:, ["Price & 1D"]])
    if "Setup Strength" in df_trades_display.columns:
        styled = styled.map(setup_strength_style, subset=IndexSlice[:, ["Setup Strength"]])

    styled = styled.set_table_styles(
        [
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ],
        overwrite=False,
    )

    column_config = build_column_config(df_trades_display.columns)

    st.dataframe(
        styled,
        width="stretch",
        height=height,
        column_config=column_config,
    )


# -------------- FETCH CORE DATA ONCE ------------------

df_tech = get_stock_summary(TOP_TECH_TICKERS)
df_ndx = get_stock_summary(NASDAQ100_TICKERS)

# -------------- EXTRA HELPERS FOR MODERN HERO KPIs ------------------


def get_row_for_ticker(df: pd.DataFrame, ticker: str):
    if df is None or df.empty:
        return None
    try:
        row = df[df["Ticker"] == ticker]
        if row.empty:
            return None
        return row.iloc[0]
    except Exception:
        return None


def render_hero_kpi(label: str, ticker: str, row: pd.Series | None, status_tuple=None):
    # Uses .hero-kpi, .hero-label, .hero-value CSS in Modern
    if ticker == "MARKET" and status_tuple is not None:
        mode, price, _, chg_pct, arrow = status_tuple
        if price is None or chg_pct is None:
            body = "<span class='hero-value'>–</span>"
            sub = ""
        else:
            body = f"<span class='hero-value'>{ticker} {arrow} {chg_pct:+.2f}%</span>"
            sub = f"<div class='hero-sub'>Spot: ${price:,.2f}</div>"
    else:
        if row is None:
            body = "<span class='hero-value'>–</span>"
            sub = ""
        else:
            price = row.get("Price", None)
            pct_1d = row.get("% 1D", None)
            pct_52 = row.get("% from 52w High", None)
            if pd.isna(price):
                body = "<span class='hero-value'>–</span>"
            else:
                if pd.isna(pct_1d):
                    body = f"<span class='hero-value'>{ticker} ${price:,.2f}</span>"
                else:
                    body = f"<span class='hero-value'>{ticker} ${price:,.2f} ({pct_1d:+.1f}%)</span>"
            if pd.isna(pct_52):
                sub = ""
            else:
                sub = f"<div class='hero-sub'>From 52w High: {pct_52:.1f}%</div>"

    html = f"""
    <div class="hero-kpi">
        <div class="hero-label">{label}</div>
        {body}
        {sub}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# -------------- LAYOUTS PER THEME ------------------


def layout_original():
    st.markdown("---")
    render_macro_section("Original")

    st.markdown("---")
    st.markdown("## Megacap & Core")

    focus_cols = st.columns([1, 1, 1])
    with focus_cols[1]:
        st.markdown(
            f"<p style='text-align:center; color:{accent}; font-weight:700; margin-bottom:0.25rem;'>Focus Table</p>",
            unsafe_allow_html=True,
        )
        focus_mode = st.checkbox(" ", key="focus_mode_orig", label_visibility="collapsed")

    render_megacap_table(df_tech, accent, focus_mode, height=600)

    st.markdown("---")
    st.markdown(
        "<h2>NASDAQ 100 Drawdown</h2>",
        unsafe_allow_html=True,
    )
    render_nasdaq_table(df_ndx, focus_mode, height=600)

    st.markdown("---")
    st.markdown("## Buy-Zone Candidates (Screened by Your Rules)")
    render_buy_zone(df_tech, df_ndx, focus_mode, height=400)

    st.markdown("---")
    st.markdown("## Trade Ideas (9-Day EMA Reclaims)")
    render_trade_ideas(df_tech, height=400)

    render_how_to()


def layout_modern():
    # Modern: brutalist cockpit with hero KPIs and hard panels
    st.markdown("---")
    st.markdown(
        """
        <div class="modern-panel">
          <div class="modern-panel-inner">
            <div style="font-size:0.7rem; letter-spacing:0.26em; text-transform:uppercase;
                        color:#9ca3af; margin-bottom:0.35rem;">
              Modern Mode · Systemic Risk Console
            </div>
            <div style="font-size:0.9rem; color:#f9fafb;">
              Core AI & semiconductor stack, filtered by your downside and valuation rules.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    focus_mode = st.checkbox(
        "Hyper-focus on AI & Semis (NVDA, TSM, AMD, AVGO, AMKR, PLTR, META)",
        key="focus_mode_modern",
        value=True,
    )

    # Hero KPIs row
    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    hero_cols = st.columns([1.1, 1, 1])

    with hero_cols[0]:
        render_hero_kpi(
            label="Market Driver (QQQ / Futures)",
            ticker="MARKET",
            row=None,
            status_tuple=status_map["MARKET"],
        )

    nvda_row = get_row_for_ticker(df_tech, "NVDA")
    tsm_row = get_row_for_ticker(df_tech, "TSM")

    with hero_cols[1]:
        render_hero_kpi(
            label="Core AI Engine",
            ticker="NVDA",
            row=nvda_row,
        )
    with hero_cols[2]:
        render_hero_kpi(
            label="Foundry Backbone",
            ticker="TSM",
            row=tsm_row,
        )

    # Main cockpit: left = Megacap, right = Buy-Zone
    st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown(
            """
            <div class="modern-panel">
              <div class="modern-panel-inner">
                <div style="font-size:0.72rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#9ca3af; margin-bottom:0.35rem;">
                  Megacap & Core
                </div>
                <div style="font-size:0.85rem; color:#e5e7eb; margin-bottom:0.45rem;">
                  Ranked by market cap with VM score and 52w damage. This is the stack that matters.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_megacap_table(df_tech, accent, focus_mode, height=430)

    with col_right:
        st.markdown(
            """
            <div class="modern-panel">
              <div class="modern-panel-inner">
                <div style="font-size:0.72rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#9ca3af; margin-bottom:0.35rem;">
                  Live Buy-Zone Screen
                </div>
                <div style="font-size:0.85rem; color:#e5e7eb; margin-bottom:0.4rem;">
                  Names that currently satisfy your drawdown and forward P/E constraints.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_buy_zone(df_tech, df_ndx, focus_mode, height=430)

    # Trade ideas row under that
    st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="modern-panel">
          <div class="modern-panel-inner">
            <div style="font-size:0.72rem; letter-spacing:0.2em; text-transform:uppercase;
                        color:#9ca3af; margin-bottom:0.35rem;">
              Trade Ideas (9-Day EMA Reclaims)
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_trade_ideas(df_tech, height=350)

    # Macro context panel at bottom
    st.markdown("<div style='margin-top:1.3rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="modern-panel">
          <div class="modern-panel-inner">
            <div style="font-size:0.72rem; letter-spacing:0.2em; text-transform:uppercase;
                        color:#9ca3af; margin-bottom:0.5rem;">
              Macro Backdrop
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_macro_section("Modern")

    # VM explainer in expander
    with st.expander("How this engine thinks (VM Score & Value Signals)"):
        render_how_to()


def layout_terminal():
    # Terminal: split-screen terminal layout
    st.markdown("---")
    col_left, col_right = st.columns([1, 1.4])

    with col_right:
        focus_mode = st.checkbox(
            "Focus tickers only (NVDA, TSM, AMD, AVGO, AMKR, PLTR, META)",
            key="focus_mode_terminal",
            value=False,
        )

    with col_left:
        st.markdown("#### Macro & Indices Monitor")
        render_macro_section("Terminal")

    with col_right:
        st.markdown("#### Megacap & Core")
        render_megacap_table(df_tech, accent, focus_mode, height=350)

        st.markdown("#### Nasdaq 100 Drawdown")
        render_nasdaq_table(df_ndx, focus_mode, height=350)

        st.markdown("#### Buy Zone Screener")
        render_buy_zone(df_tech, df_ndx, focus_mode, height=260)

        st.markdown("#### Trade Ideas (9 Day EMA Reclaims)")
        render_trade_ideas(df_tech, height=260)
    # Terminal mode = pure screen, no explanation block


# -------------- CHOOSE LAYOUT ------------------

if current_theme == "Original":
    layout_original()
elif current_theme == "Modern":
    layout_modern()
else:  # Terminal
    layout_terminal()
