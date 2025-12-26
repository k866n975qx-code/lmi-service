from typing import Optional
import pandas as pd
from .common import normalize_prices


class YFinanceAdapter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        try:
            if enabled:
                import yfinance as yf  # type: ignore
                self.yf = yf
            else:
                self.yf = None
        except Exception:
            self.enabled = False
            self.yf = None

    def prices(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.enabled or self.yf is None:
            return None
        try:
            df = self.yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
            if isinstance(df, pd.DataFrame):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df.reset_index()
                return normalize_prices(df, symbol)
        except Exception:
            return None
        return None

    def prices_batch(self, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        if not self.enabled or self.yf is None:
            return {}
        if not symbols:
            return {}
        try:
            tickers = " ".join(symbols)
            df = self.yf.download(
                tickers,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
            )
            if df is None or getattr(df, "empty", True):
                return {}
            out: dict[str, pd.DataFrame] = {}
            if isinstance(df.columns, pd.MultiIndex):
                level0 = df.columns.get_level_values(0)
                level1 = df.columns.get_level_values(1)
                if any(sym in level0 for sym in symbols):
                    for sym in symbols:
                        if sym not in level0:
                            continue
                        sdf = df[sym].reset_index()
                        norm = normalize_prices(sdf, sym)
                        if norm is not None and not norm.empty:
                            out[sym] = norm
                elif any(sym in level1 for sym in symbols):
                    for sym in symbols:
                        if sym not in level1:
                            continue
                        sdf = df.xs(sym, level=1, axis=1).reset_index()
                        norm = normalize_prices(sdf, sym)
                        if norm is not None and not norm.empty:
                            out[sym] = norm
                return out
            if len(symbols) == 1:
                sdf = df.reset_index()
                norm = normalize_prices(sdf, symbols[0])
                if norm is not None and not norm.empty:
                    out[symbols[0]] = norm
            return out
        except Exception:
            return {}

    def dividends(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.enabled or self.yf is None:
            return None
        try:
            t = self.yf.Ticker(symbol)
            dv = t.dividends
            if isinstance(dv, pd.Series):
                return dv.to_frame("dividend").reset_index()
            if isinstance(dv, pd.DataFrame):
                return dv.reset_index()
        except Exception:
            return None
        return None

    def splits(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.enabled or self.yf is None:
            return None
        try:
            t = self.yf.Ticker(symbol)
            sp = t.splits
            if isinstance(sp, pd.Series):
                return sp.to_frame("split").reset_index()
            if isinstance(sp, pd.DataFrame):
                return sp.reset_index()
        except Exception:
            return None
        return None
