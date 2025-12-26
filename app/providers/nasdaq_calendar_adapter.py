from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx
import pandas as pd

from ..config import settings
from ..utils import retry_call

class NasdaqCalendarAdapter:
    def __init__(self, enabled: bool = True, timeout: float = 10.0):
        self.enabled = enabled
        self.timeout = timeout
        self.headers = {
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36",
            "origin": "https://www.nasdaq.com",
            "referer": "https://www.nasdaq.com/",
        }

    def _parse_date(self, val: str | None):
        if not val:
            return None
        text = str(val).strip()
        if not text or text.upper() == "N/A":
            return None
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None

    def _parse_amount(self, val: str | None):
        if val is None:
            return None
        text = str(val).strip()
        if not text or text.upper() == "N/A":
            return None
        text = text.replace("$", "").replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    def _fetch_dividends(self, symbol: str, assetclass: str):
        url = f"https://api.nasdaq.com/api/quote/{symbol}/dividends"
        def _call():
            resp = httpx.get(
                url,
                headers=self.headers,
                params={"assetclass": assetclass},
                timeout=min(self.timeout, settings.http_timeout_seconds),
            )
            if resp.status_code != 200:
                raise RuntimeError(f"nasdaq_status_{resp.status_code}")
            return resp.json()
        try:
            payload = retry_call(
                _call,
                attempts=settings.http_retry_attempts,
                base_delay=settings.http_retry_backoff_seconds,
            )
        except Exception:
            return []
        rows = (payload.get("data") or {}).get("dividends", {}).get("rows", [])
        return rows or []

    def dividends(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.enabled:
            return None
        rows = self._fetch_dividends(symbol, "stocks")
        if not rows:
            rows = self._fetch_dividends(symbol, "etf")
        if not rows:
            return None
        out = []
        for row in rows:
            ex_date = self._parse_date(row.get("exOrEffDate"))
            pay_date = self._parse_date(row.get("paymentDate"))
            amount = self._parse_amount(row.get("amount"))
            if ex_date is None or amount is None:
                continue
            out.append(
                {
                    "ex_date": ex_date.isoformat(),
                    "pay_date": pay_date.isoformat() if pay_date else None,
                    "amount": amount,
                    "currency": row.get("currency"),
                }
            )
        if not out:
            return None
        return pd.DataFrame(out)
