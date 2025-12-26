from typing import Optional
import pandas as pd
from .common import normalize_prices

class OpenBBAdapter:
    def __init__(self, enabled: bool=True):
        self.enabled = enabled
        try:
            if enabled:
                from openbb import obb  # type: ignore
                self.obb = obb
            else:
                self.obb = None
        except Exception:
            self.obb = None
            self.enabled = False

    def prices(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.enabled or self.obb is None: return None
        try:
            df = self.obb.equity.price.historical(symbol, provider="yfinance", start=start, end=end, interval="1d")
            return normalize_prices(df, symbol)
        except Exception:
            return None

    def dividends(self, symbol: str):
        if not self.enabled or self.obb is None: return None
        try:
            return self.obb.equity.dividends(symbol, provider="yfinance")
        except Exception:
            return None

    def splits(self, symbol: str):
        if not self.enabled or self.obb is None: return None
        try:
            return self.obb.equity.splits(symbol, provider="yfinance")
        except Exception:
            return None
