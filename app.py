import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

# -------------- CONFIG ------------------
st.set_page_config(page_title="Tech Snapshot", layout="wide")
st.title("🔍 AI, Infrastructure, Network, Supply chain")
st.caption("Last updated: {}".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))

# ✅ Manually ordered by market cap / your preference
TOP_TECH_TICKERS = [
    "MSFT", "AMZN", "GOOG", "NVDA", "META",
    "TSM", "AVGO", "ORCL", "CRM",
    "AMD", "NOW", "MU", "SNOW", "PLTR",
    "ANET", "CRWD", "PANW", "NET", "DDOG",
    "MDB", "MRVL", "IBM", "AMKR", "SMCI"
]


def get_value_momentum_signal(rsi, pct_from_high, pct_1m, fpe):
    """
    Combined value + momentum signal based on:
    - RSI
    - % from 52-week high
    - 1-month performance
    - Forward P/E
    """
    if rsi is None or pct_from_high is None:
        return "❔ Check data"

    # Deep value pullback
    if rsi < 35 and pct_from_high <= -30 and (fpe is not None and fpe <= 30):
        return "💚 Deep value pullback"

    # Value-ish pullback
    if rsi < 50 and pct_from_high <= -15 and (fpe is not None and fpe <= 35):
        return "🟡 Value watch"

    # Strong momentum trend
    if 50 <= rsi <= 70 and (pct_1m is not None and pct_1m > 0):
        return "🔵 Momentum trend"

    # Overheated / extended
    if rsi > 70 or pct_from_high >= -5 or (fpe is not None and fpe >= 45):
        return "🔴 Hot / extended"

    # Everything else
    return "⚪ Neutral"


# -------------- FETCH FUNCTION ------------------
@st.cache_data(ttl=600)
def get_stock_summary(tickers):
    rows = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # 🔄 1-year history for 52-week logic
            hist = stock.history(period="1y")
            info = stock.info

            if hist.empty or "Close" not in hist.columns:
                st.warning(f"⚠️ No price history for {ticker}, skipping.")
                continue

            close = hist["Close"]

            if len(close) < 10:
                st.warning(f"⚠️ Not enough data points for {ticker}, skipping.")
                continue

            price = close.iloc[-1]

            pct_5d = (
                round((price - close.iloc[-6]) / close.iloc[-6] * 100, 2)
                if len(close) >= 6 else None
            )
            pct_1m = (
                round((price - close.iloc[-22]) / close.iloc[-22] * 100, 2)
                if len(close) >= 22 else None
            )

            # 🔥 52-week high logic
            high_52wk = close.max()
            pct_from_52wk = round((price - high_52wk) / high_52wk * 100, 2)

            # RSI
            rsi = RSIIndicator(close=close).rsi()
            rsi_val = round(rsi.iloc[-1], 2)

            # Valuation
            pe = info.get("trailingPE", None)
            fpe = info.get("forwardPE", None)

            # Simple RSI label (still useful as a quick read)
            rsi_signal = (
                "💚 Oversold" if rsi_val < 30 else
                "🟡 Watch" if rsi_val < 50 else
                "🔵 Trend" if rsi_val < 70 else
                "🔴 Overbought"
            )

            # Combined value + momentum signal
            value_signal = get_value_momentum_signal(
                rsi=rsi_val,
                pct_from_high=pct_from_52wk,
                pct_1m=pct_1m,
                fpe=fpe
            )

            rows.append({
                "Ticker": ticker,
                "Price": f"${price:.2f}",
                "% 5D": f"{pct_5d:.1f}%" if pct_5d is not None else "–",
                "% 1M": f"{pct_1m:.1f}%" if pct_1m is not None else "–",
                "% from 52w High": f"{pct_from_52wk:.1f}%" if pct_from_52wk is not None else "–",
                "RSI": f"{rsi_val:.1f}",
                "RSI Zone": rsi_signal,
                "Value Signal": value_signal,
                "P/E": f"{pe:.1f}" if pe is not None else "–",
                "Fwd P/E": f"{fpe:.1f}" if fpe is not None else "–",
            })

        except Exception as e:
            st.warning(f"⚠️ Skipping {ticker} due to error: {e}")

    return pd.DataFrame(rows)


# -------------- MAIN DISPLAY ------------------
with st.spinner("📡 Fetching data..."):
    df = get_stock_summary(TOP_TECH_TICKERS)

df["Ticker"] = pd.Categorical(df["Ticker"], categories=TOP_TECH_TICKERS, ordered=True)
df = df.sort_values("Ticker")

st.subheader("📊 Pullback & Momentum Overview")
st.dataframe(df, use_container_width=True)

# -------------- FOOTER ------------------
st.markdown("---")
st.markdown("""
### 📘 How to read the signals

**RSI Zone (classic RSI view)**  
- 💚 **Buy** = RSI < 30 (oversold)  
- 🟡 **Watch** = 30–50 (weak, potential base)  
- 🔵 **Trend** = 50–70 (healthy uptrend)  
- 🔴 **Overbought** = RSI > 70 (stretched short term)  

**Value Signal (combined value + momentum)**  
- 💚 **Deep value pullback** – RSI washed out, price far below its 52-week high, and forward P/E not extreme.  
- 🟡 **Value watch** – Decent pullback and weak RSI, with reasonable forward P/E.  
- 🔵 **Momentum trend** – Positive 1M performance with mid-range RSI (50–70).  
- 🔴 **Hot / extended** – Near 52-week highs and/or expensive on forward P/E, or overbought RSI.  
- ⚪ **Neutral** – No strong value or momentum pattern detected.

**📈 Idea:** Focus on 💚 / 🟡 names for pullbacks and 🔵 for trend-following. Treat 🔴 as caution / trim zone.
""")
