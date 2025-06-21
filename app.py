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

            # --- % changes ---
            pct_5d = round((price - close.iloc[-6]) / close.iloc[-6] * 100, 2) if len(close) >= 6 else None
            pct_1m = round((price - close.iloc[-22]) / close.iloc[-22] * 100, 2) if len(close) >= 22 else None
            high_2yr = close.max()
            pct_from_ath = round((price - high_2yr) / high_2yr * 100, 2)

            # RSI
            rsi = RSIIndicator(close=close).rsi()
            rsi_val = round(rsi.iloc[-1], 2)

            # Fundamentals
            pe = info.get("trailingPE", None)
            fpe = info.get("forwardPE", None)
            pb = info.get("priceToBook", None)

            # RSI Signal
            rsi_signal = (
                "💚 Buy" if rsi_val < 30 else
                "🟡 Watch" if rsi_val < 50 else
                "🔵 Trend" if rsi_val < 70 else
                "🔴 Overbought"
            )

            rows.append({
                "Ticker": ticker,
                "Price": price,
                "% 5D": pct_5d,
                "% 1M": pct_1m,
                "% from ATH": pct_from_ath,
                "RSI": rsi_val,
                "RSI Signal": rsi_signal,
                "P/E": pe,
                "Fwd P/E": fpe,
                "P/B": pb,
            })

        except Exception as e:
            st.warning(f"⚠️ Skipping {ticker} due to error: {e}")

    return pd.DataFrame(rows)

# -------------- MAIN DISPLAY ------------------
with st.spinner("📡 Fetching data..."):
    df = get_stock_summary(TOP_TECH_TICKERS)

# ✅ Preserve your manual market cap order
df["Ticker"] = pd.Categorical(df["Ticker"], categories=TOP_TECH_TICKERS, ordered=True)
df = df.sort_values("Ticker")

st.subheader("📊 Pullback & Momentum Overview")

# Styling
styled_df = df.style \
    .set_properties(**{'text-align': 'center'}) \
    .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]) \
    .background_gradient(subset=["% 5D", "% 1M", "% from ATH"], cmap="YlOrRd", low=0.2, high=0.8) \
    .background_gradient(subset=["RSI"], cmap="RdYlGn_r", low=0.2, high=0.8) \
    .format({
        "Price": "${:.2f}",
        "% 5D": "{:.1f}%",
        "% 1M": "{:.1f}%",
        "% from ATH": "{:.1f}%",
        "RSI": "{:.1f}",
        "P/E": "{:.1f}",
        "Fwd P/E": "{:.1f}",
        "P/B": "{:.1f}"
    }, na_rep="–")

st.dataframe(styled_df, use_container_width=True)

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
