from typing import Optional
import pandas as pd
from .common import normalize_prices

class YahooQueryAdapter:
    def __init__(self, enabled: bool=True):
        self.enabled = enabled
        try:
            if enabled:
                import yahooquery as yq  # type: ignore
                self.yq = yq
            else:
                self.yq = None
        except Exception:
            self.enabled = False
            self.yq = None

    def prices(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.enabled or self.yq is None: return None
        try:
            t = self.yq.Ticker(symbol)
            df = t.history(start=start, end=end, adj_ohlc=True)
            if isinstance(df, pd.DataFrame):
                if 'symbol' not in df.columns:
                    df = df.reset_index()
                return normalize_prices(df, symbol)
        except Exception:
            return None
        return None

    def prices_batch(self, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        if not self.enabled or self.yq is None:
            return {}
        if not symbols:
            return {}
        try:
            t = self.yq.Ticker(symbols)
            df = t.history(start=start, end=end, adj_ohlc=True)
            if df is None or getattr(df, "empty", True):
                return {}
            if "symbol" not in df.columns:
                df = df.reset_index()
            out: dict[str, pd.DataFrame] = {}
            if "symbol" in df.columns:
                for sym in symbols:
                    sdf = df[df["symbol"] == sym]
                    if sdf.empty:
                        continue
                    norm = normalize_prices(sdf, sym)
                    if norm is not None and not norm.empty:
                        out[sym] = norm
            return out
        except Exception:
            return {}

    def dividends(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.enabled or self.yq is None: return None
        try:
            t = self.yq.Ticker(symbol)
            dv = t.dividends
            if isinstance(dv, pd.DataFrame):
                return dv.reset_index()
        except Exception:
            return None
        return None

    def splits(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.enabled or self.yq is None: return None
        try:
            t = self.yq.Ticker(symbol)
            sp = t.splits
            if isinstance(sp, pd.DataFrame):
                return sp.reset_index()
        except Exception:
            return None
        return None
