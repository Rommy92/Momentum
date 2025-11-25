import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

# -------------- CONFIG ------------------
st.set_page_config(page_title="Tech Snapshot", layout="wide")
st.title("🔍 Dashboard - AI, Infra, SaaS, Cybersecurity")
st.caption("Last updated: {}".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))

# ✅ Manually ordered by market cap descending
TOP_TECH_TICKERS = [
    "MSFT", "AMZN", "GOOG", "NVDA", "META",
    "TSM", "AVGO", "ORCL", "NFLX", "CRM",
    "AMD", "NOW", "MU", "SNOW", "PLTR", "ANET", "CRWD", "PANW", "NET", "DDOG", "MDB", "MRVL"
]


def get_value_momentum_signal(rsi, pct_from_ath, pct_1m, fpe):
    # If critical data missing, bail out
    if rsi is None or pct_from_ath is None:
        return "❔ Check data"

    # Deep value pullback
    if rsi < 35 and pct_from_ath <= -25 and (fpe is not None and fpe <= 30):
        return "💚 Deep value pullback"

    # Value-ish pullback
    if rsi < 50 and pct_from_ath <= -15 and (fpe is not None and fpe <= 35):
        return "🟡 Value watch"

    # Strong momentum trend
    if 50 <= rsi <= 70 and (pct_1m is not None and pct_1m > 0):
        return "🔵 Momentum trend"

    # Overheated / extended
    if rsi > 70 or pct_from_ath >= -5 or (fpe is not None and fpe >= 45):
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
            hist = stock.history(period="2y")
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

            high_2yr = close.max()
            pct_from_ath = round((price - high_2yr) / high_2yr * 100, 2)

            rsi = RSIIndicator(close=close).rsi()
            rsi_val = round(rsi.iloc[-1], 2)

            pe = info.get("trailingPE", None)
            fpe = info.get("forwardPE", None)

            rsi_signal = (
                "💚 Buy" if rsi_val < 30 else
                "🟡 Watch" if rsi_val < 50 else
                "🔵 Trend" if rsi_val < 70 else
                "🔴 Overbought"
            )

            value_signal = get_value_momentum_signal(
                rsi=rsi_val,
                pct_from_ath=pct_from_ath,
                pct_1m=pct_1m,
                fpe=fpe
            )

            rows.append({
                "Ticker": ticker,
                "Price": f"${price:.2f}",
                "% 5D": f"{pct_5d:.1f}%" if pct_5d is not None else "–",
                "% 1M": f"{pct_1m:.1f}%" if pct_1m is not None else "–",
                "% from ATH": f"{pct_from_ath:.1f}%" if pct_from_ath is not None else "–",
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
### 📘 Value & Momentum Signals (How the algorithm ranks each stock)

**💚 Deep value pullback**  
Triggered when RSI is washed out, price is far below its 2-year high, and forward valuation is reasonable.  
This combination generally marks a high-quality dip.

**🟡 Value watch**  
Moderate pullback with improving conditions.  
Price is below recent highs and RSI is weak, but valuation remains acceptable.

**🔵 Momentum trend**  
Strong, healthy trend.  
RSI in mid-range (50–70) and recent performance is positive.

**🔴 Hot / extended**  
RSI overbought or valuation stretched, or price sitting near 2-year highs.  
Often signals a period of consolidation or mean reversion.

**⚪ Neutral**  
No strong value or momentum signal detected.  
Stock is in the middle of its normal range.
""")

