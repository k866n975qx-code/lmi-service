from io import StringIO
from typing import Optional
import json
import pandas as pd
import urllib.parse
import urllib.request

class FredAdapter:
    def __init__(self, api_key: str | None, timeout_seconds: float = 15.0):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        try:
            import pyfredapi as pf  # type: ignore
            self.fred = pf
        except Exception:
            self.fred = None

    def _fetch_json_series(self, series_id: str) -> Optional[pd.DataFrame]:
        if not self.api_key:
            return None
        try:
            query = urllib.parse.urlencode(
                {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                }
            )
            req = urllib.request.Request(
                f"https://api.stlouisfed.org/fred/series/observations?{query}",
                headers={"User-Agent": "lmi-service/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            observations = payload.get("observations") or []
            if not observations:
                return None
            rows = []
            for item in observations:
                rows.append(
                    {
                        "date": item.get("date"),
                        "value": pd.to_numeric(item.get("value"), errors="coerce"),
                    }
                )
            df = pd.DataFrame(rows)
            if df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            return df.dropna(subset=["date"])
        except Exception:
            return None

    def _fetch_csv_series(self, series_id: str) -> Optional[pd.DataFrame]:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(series_id)}"
            req = urllib.request.Request(url, headers={"User-Agent": "lmi-service/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                text = resp.read().decode("utf-8")
            df = pd.read_csv(StringIO(text))
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

    def series(self, series_id: str) -> Optional[pd.DataFrame]:
        if self.fred is not None:
            try:
                client = self.fred.Fred(series_id=series_id, api_key=self.api_key) if self.api_key else self.fred.Fred(series_id=series_id)
                data = client.series.observations.to_pandas()
                return data
            except Exception:
                pass
        api_df = self._fetch_json_series(series_id)
        if api_df is not None:
            return api_df
        return self._fetch_csv_series(series_id)
