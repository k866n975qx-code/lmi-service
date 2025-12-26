from typing import Optional
import pandas as pd
import pandas_datareader.data as web
from .common import normalize_prices

class StooqAdapter:
    def __init__(self, enabled: bool=True):
        self.enabled = enabled

    def prices(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.enabled: return None
        try:
            df = web.DataReader(symbol, 'stooq', start=start, end=end)
            if isinstance(df, pd.DataFrame):
                df = df.reset_index().rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'})
                out = normalize_prices(df, symbol)
                if out is not None and not out.empty:
                    return out
        except Exception:
            pass
        # Fallback to stooq CSV endpoint (no API key)
        try:
            stooq_symbol = symbol
            if "." not in stooq_symbol:
                stooq_symbol = f"{stooq_symbol}.US"
            url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
            df = pd.read_csv(url)
            if isinstance(df, pd.DataFrame):
                df = df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
                return normalize_prices(df, symbol)
        except Exception:
            return None
        return None
