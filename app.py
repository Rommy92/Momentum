import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

# -------------- CONFIG ------------------
st.set_page_config(page_title="Tech Snapshot", layout="wide")
st.title("🔍 AI, Infrastructure, Network, Supply chain")
st.caption("Last updated: {}".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))

# Your tickers
TOP_TECH_TICKERS = [
    "MSFT", "AMZN", "GOOG", "NVDA", "META",
    "TSM", "AVGO", "ORCL", "CRM",
    "AMD", "NOW", "MU", "SNOW", "PLTR",
    "ANET", "CRWD", "PANW", "NET", "DDOG",
    "MDB", "MRVL", "IBM", "AMKR", "SMCI"
]


def get_value_momentum_signal(rsi, pct_from_high, pct_1m, fpe):
    if rsi is None or pct_from_high is None:
        return "❔ Check data"

    if rsi < 35 and pct_from_high <= -30 and (fpe is not None and fpe <= 30):
        return "💚 Deep value pullback"

    if rsi < 50 and pct_from_high <= -15 and (fpe is not None and fpe <= 35):
        return "🟡 Value watch"

    if 50 <= rsi <= 70 and (pct_1m is not None and pct_1m > 0):
        return "🔵 Momentum trend"

    if rsi > 70 or pct_from_high >= -5 or (fpe is not None and fpe >= 45):
        return "🔴 Hot / extended"

    return "⚪ Neutral"


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

            # ---- Fundamentals ----
            try:
                info = stock.get_info()
            except Exception:
                info = {}

            pe = info.get("trailingPE", None)
            try:
                pe = float(pe)
            except:
                pe = None

            # ----- Forward EPS (get_eps_trend) -----
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
            except:
                forward_eps = None

            # fallback
            if forward_eps is None:
                try:
                    forward_eps = float(info.get("forwardEps"))
                except:
                    forward_eps = None

            if forward_eps and forward_eps > 0:
                fpe = round(price / forward_eps, 2)

            # Market Cap
            market_cap = info.get("marketCap", None)

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
                "Market Cap": market_cap,
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

        except Exception:
            continue

    return pd.DataFrame(rows)


# -------------- MAIN DISPLAY ------------------
with st.spinner("📡 Fetching data..."):
    df = get_stock_summary(TOP_TECH_TICKERS)

# Sort by Market Cap DESC
df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")
df = df.sort_values("Market Cap", ascending=False)

st.subheader("📊 Pullback & Momentum Overview")
st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### How to read signals...
""")
