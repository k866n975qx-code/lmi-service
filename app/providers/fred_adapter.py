from typing import Optional
import pandas as pd
import urllib.parse

class FredAdapter:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        try:
            import pyfredapi as pf  # type: ignore
            self.fred = pf
        except Exception:
            self.fred = None

    def series(self, series_id: str) -> Optional[pd.DataFrame]:
        if self.fred is not None:
            try:
                client = self.fred.Fred(series_id=series_id, api_key=self.api_key) if self.api_key else self.fred.Fred(series_id=series_id)
                data = client.series.observations.to_pandas()
                return data
            except Exception:
                pass
        # Fallback: keyless CSV from fredgraph
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(series_id)}"
            df = pd.read_csv(url)
            if df.empty:
                return None
            if "DATE" in df.columns:
                df = df.rename(columns={"DATE": "date"})
            if series_id in df.columns:
                df = df.rename(columns={series_id: "value"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df
        except Exception:
            return None
