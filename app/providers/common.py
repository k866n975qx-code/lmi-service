import pandas as pd

CANON_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]

def normalize_prices(df, symbol: str):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    d = df.copy()
    if "date" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={"index": "date"})
        elif "Datetime" in d.columns:
            d = d.rename(columns={"Datetime": "date"})
    rename = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "adjclose": "adj_close",
        "AdjClose": "adj_close",
        "Volume": "volume",
        "Symbol": "symbol",
    }
    d = d.rename(columns=rename)
    if "adj_close" not in d.columns and "close" in d.columns:
        d["adj_close"] = d["close"]
    if "symbol" not in d.columns:
        d["symbol"] = symbol
    keep = [c for c in CANON_COLS if c in d.columns]
    d = d[keep]
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"]).dt.date
    for c in ["open", "high", "low", "close", "adj_close"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "volume" in d.columns:
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0).astype("int64")
    return d.sort_values("date").reset_index(drop=True)
