import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

# -------------- CONFIG ------------------
st.set_page_config(page_title="Tech Snapshot", layout="wide")
st.title("🔍 Tech Stocks Snapshot Dashboard (Ranked by Market Cap)")
st.caption("Last updated: {}".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))

# ✅ Manually ordered by market cap descending
TOP_TECH_TICKERS = [
    "MSFT", "AMZN", "GOOG", "NVDA", "META",
    "TSM", "AVGO", "ORCL", "NFLX", "CRM",
    "AMD", "NOW", "MU", "SNOW", "PLTR"
]

# -------------- FETCH FUNCTION ------------------
@st.cache_data(ttl=600)
def get_stock_summary(tickers):
    rows = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            info = stock.info

            close = hist['Close']
            price = close.iloc[-1]

            pct_5d = round((price - close.iloc[-6]) / close.iloc[-6] * 100, 2) if len(close) >= 6 else None
            pct_1m = round((price - close.iloc[-22]) / close.iloc[-22] * 100, 2) if len(close) >= 22 else None
            high_2yr = close.max()
            pct_from_ath = round((price - high_2yr) / high_2yr * 100, 2)

            rsi = RSIIndicator(close=close).rsi()
            rsi_val = round(rsi.iloc[-1], 2)

            pe = info.get("trailingPE", None)
            fpe = info.get("forwardPE", None)
            pb = info.get("priceToBook", None)

            rsi_signal = (
                "💚 Buy" if rsi_val < 30 else
                "🟡 Watch" if rsi_val < 50 else
                "🔵 Trend" if rsi_val < 70 else
                "🔴 Overbought"
            )

            rows.append({
                "Ticker": ticker,
                "Price": f"${price:.2f}",
                "% 5D": f"{pct_5d:.1f}%" if pct_5d is not None else "–",
                "% 1M": f"{pct_1m:.1f}%" if pct_1m is not None else "–",
                "% from ATH": f"{pct_from_ath:.1f}%" if pct_from_ath is not None else "–",
                "RSI": f"{rsi_val:.1f}",
                "RSI Signal": rsi_signal,
                "P/E": f"{pe:.1f}" if pe is not None else "–",
                "Fwd P/E": f"{fpe:.1f}" if fpe is not None else "–",
                "P/B": f"{pb:.1f}" if pb is not None else "–",
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
**📘 RSI Signal Zones:**  
- 💚 **Buy** = RSI < 30  
- 🟡 **Watch** = 30–50  
- 🔵 **Trend** = 50–70  
- 🔴 **Overbought** = RSI > 70  

**📈 Strategy Tip:** Combine **RSI < 30** with **% from ATH < –20%** and **down 1M** to signal a margin-ready pullback.
""")
