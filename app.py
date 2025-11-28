@st.cache_data(ttl=60)
def get_ticker_status(symbol: str):
    """
    Realtime-like status for a ticker.

    - If marketState == PRE  -> use preMarketPrice vs regularMarketPreviousClose
    - If marketState == POST -> use postMarketPrice vs regularMarketPreviousClose
    - Else                   -> use regularMarketPrice vs regularMarketPreviousClose
    - If that fails, fall back to 2d history + 1m intraday.

    Returns (mode, price, change, change_pct, arrow).
    """
    # ---------- PATH 1: use Yahoo info() ----------
    try:
        t = yf.Ticker(symbol)
        info = t.get_info() or {}
    except Exception:
        info = {}

    def build_from_info(price_key: str, state: str):
        """Helper to compute price/change/% using info dict."""
        price = info.get(price_key, None)
        prev_close = info.get("regularMarketPreviousClose", None)

        if price is None or prev_close is None:
            return None

        try:
            price = float(price)
            prev_close = float(prev_close)
        except Exception:
            return None

        if prev_close == 0:
            return None

        change = price - prev_close
        change_pct = (change / prev_close) * 100.0

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

    # Try to use the appropriate extended-hours field based on marketState
    state = str(info.get("marketState", "")).upper()

    # Pre-market price (matches Yahoo's "Pre-Market" line)
    if state == "PRE":
        res = build_from_info("preMarketPrice", state="PRE")
        if res is not None:
            return res

    # Post-market price (matches Yahoo's "Post-Market" line)
    if state == "POST":
        res = build_from_info("postMarketPrice", state="POST")
        if res is not None:
            return res

    # Regular session
    res = build_from_info("regularMarketPrice", state="REGULAR")
    if res is not None:
        return res

    # ---------- PATH 2: fallback to 2d daily + 1m intraday ----------
    try:
        t = yf.Ticker(symbol)

        # 2 most recent daily closes
        daily = t.history(period="2d")
        closes = daily.get("Close", pd.Series(dtype=float)).dropna()
        if len(closes) == 0:
            return "neutral", None, None, None, "▶"

        if len(closes) >= 2:
            prev_close = float(closes.iloc[-2])
        else:
            prev_close = float(closes.iloc[-1])

        # Last intraday trade including pre/post
        intra = t.history(period="1d", interval="1m", prepost=True)
        intra_closes = intra.get("Close", pd.Series(dtype=float)).dropna()
        if len(intra_closes) > 0:
            price = float(intra_closes.iloc[-1])
        else:
            price = prev_close

        change = price - prev_close
        change_pct = (change / prev_close) * 100.0 if prev_close != 0 else 0.0

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
