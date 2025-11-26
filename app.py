import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from matplotlib.colors import LinearSegmentedColormap

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

/* Buttons */
.stButton>button {{
    background-color: #000000 !important;
    border: 1px solid {accent} !important;
    color: {accent} !important;
    border-radius: 6px;
    padding: 0.45rem 1.1rem;
    box-shadow: 0 0 12px {accent}aa;
}}
.stButton>button:hover {{
    background-color: {accent}22 !important;
    box-shadow: 0 0 18px {accent};
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


# -------------- TICKERS ------------------

TOP_TECH_TICKERS = [
    "MSFT", "AMZN", "GOOG", "NVDA", "META",
    "TSM", "AVGO", "ORCL", "CRM",
    "AMD", "NOW", "MU", "SNOW", "PLTR",
    "ANET", "CRWD", "PANW", "NET", "DDOG",
    "MDB", "MRVL", "IBM", "AMKR", "SMCI",
    "AXON", "ISRG",   # INTU -> ISRG
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


def rsi_style(val):
    """CSS style for extreme RSI values."""
    if pd.isna(val):
        return ""
    v = float(val)
    if v < 30:
        return "color: #22c55e; font-weight: 600;"  # green
    if v > 70:
        return "color: #ef4444; font-weight: 600;"  # red
    return ""


# Red → black → green colormap for heatmaps
heatmap_cmap = LinearSegmentedColormap.from_list(
    "rbkgn", ["#ef4444", "#000000", "#22c55e"]
)


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

            rsi_signal = (
                "💚 Oversold" if rsi_val < 30 else
                "🟡 Watch" if rsi_val < 50 else
                "🔵 Trend" if rsi_val < 70 else
                "🔴 Overbought"
            )

            value_signal = get_value_momentum_signal(
                rsi=rsi_val,
                pct_from_high=pct_from_52wk,
                pct_1m=pct_1m,
                fpe=fpe
            )

            rows.append({
                "Ticker": ticker,
                "Price": price,
                "% 5D": pct_5d,
                "% 1M": pct_1m,
                "% from 52w High": pct_from_52wk,
                "RSI": rsi_val,
                "RSI Zone": rsi_signal,
                "Value Signal": value_signal,
                "P/E": pe,
                "Fwd P/E": fpe,
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


# -------------- MAIN DISPLAY ------------------

with st.spinner("📡 Fetching data..."):
    df = get_stock_summary(TOP_TECH_TICKERS)

if not df.empty:
    # Use ticker as index — index column is effectively "frozen" when horizontal scrolling
    df = df.set_index("Ticker")

    # Formatting (keeps underlying data numeric)
    format_dict = {
        "Price": "${:,.2f}",
        "% 5D": "{:.1f}%",
        "% 1M": "{:.1f}%",
        "% from 52w High": "{:.1f}%",
        "RSI": "{:.1f}",
        "P/E": "{:.1f}",
        "Fwd P/E": "{:.1f}",
    }

    styled = df.style.format(format_dict, na_rep="–")

    # ---- Heatmaps ----
    pct_cols = ["% 5D", "% 1M"]
    dist_col = "% from 52w High"

    # Symmetric range around 0 for % 5D / % 1M
    max_abs_change = max(
        abs(df[pct_cols].min().min()),
        abs(df[pct_cols].max().max())
    )
    styled = styled.background_gradient(
        cmap=heatmap_cmap,
        subset=pct_cols,
        vmin=-max_abs_change,
        vmax=max_abs_change,
    )

    # 52w distance: values typically <= 0. 0% (near highs) should be green, deep negatives red.
    if df[dist_col].notna().any():
        min_drawdown = df[dist_col].min()  # most negative
        styled = styled.background_gradient(
            cmap=heatmap_cmap,
            subset=[dist_col],
            vmin=min_drawdown,
            vmax=0,
        )

    # Right-align numeric columns and center headers
    numeric_cols = ["Price", "% 5D", "% 1M", "% from 52w High", "RSI", "P/E", "Fwd P/E"]

    styled = (
        styled.set_table_styles(
            [{"selector": "th.col_heading", "props": [("text-align", "center")]}],
            overwrite=False,
        )
        .set_properties(subset=numeric_cols, **{"text-align": "right"})
    )

    # RSI highlight (<30 green, >70 red)
    styled = styled.applymap(rsi_style, subset=["RSI"])

    # Autosize columns like clicking "Autosize" in the UI
    column_config = {
        col: st.column_config.Column(width="fit") for col in df.columns
    }

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        column_config=column_config,
    )
else:
    st.write("No data loaded.")

st.markdown("---")
st.markdown("""
### 📘 How to read the signals

**RSI Zone (classic RSI view)**  
- 💚 **Oversold** = RSI < 30  
- 🟡 **Watch** = 30–50  
- 🔵 **Trend** = 50–70  
- 🔴 **Overbought** = RSI > 70  

**Value Signal (combined value + momentum)**  
- 💚 **Deep value pullback** – RSI washed out, far below 52-week high, reasonable forward P/E.  
- 🟡 **Value watch** – Decent pullback + weak RSI, forward P/E not extreme.  
- 🔵 **Momentum trend** – Positive 1M performance, RSI 50–70.  
- 🔴 **Hot / extended** – Near highs and/or expensive P/E, or overbought RSI.  
- ⚪ **Neutral** – No strong pattern.
""")
