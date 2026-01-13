from typing import List, Dict, Any
import time
from datetime import datetime, timezone
import pandas as pd
from ..config import settings
from ..utils import retry_call, RateLimiter, now_utc_iso
from ..providers.yfinance_adapter import YFinanceAdapter
from ..providers.yahooquery_adapter import YahooQueryAdapter
from ..providers.stooq_adapter import StooqAdapter
from ..providers.openbb_adapter import OpenBBAdapter
from ..providers.fred_adapter import FredAdapter
from ..cache_layer import CacheLayer

def _df_to_records(df: pd.DataFrame):
    if df is None or getattr(df, "empty", True):
        return None
    data = df.copy()
    if "date" in data.columns:
        data["date"] = data["date"].astype(str)
    return data.to_dict(orient="records")

def _records_to_df(records):
    if not records:
        return None
    df = pd.DataFrame.from_records(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df

def _chunked(items, size: int):
    if size <= 0:
        size = 1
    for idx in range(0, len(items), size):
        yield items[idx: idx + size]

def _df_is_usable(df):
    return df is not None and not getattr(df, "empty", True)

def _retry_on_empty_df(df):
    if df is None:
        return True
    if getattr(df, "empty", False):
        return True
    return False

class MarketData:
    def __init__(self):
        self.providers = {
            'yfinance': YFinanceAdapter(enabled=bool(settings.yf_enable)),
            'openbb': OpenBBAdapter(enabled=bool(settings.providers_openbb)),
            'yq': YahooQueryAdapter(enabled=bool(settings.yq_enable)),
            'stooq': StooqAdapter(enabled=bool(settings.stooq_enable)),
        }
        self.fred = FredAdapter(api_key=settings.fred_api_key)
        self.prices: Dict[str, Any] = {}
        self.quotes: Dict[str, Any] = {}
        self.provenance: Dict[str, list] = {}
        self.cache = CacheLayer(
            settings.cache_dir,
            settings.cache_db_path,
            settings.cache_ttl_hours,
        ) if bool(settings.cache_enabled) else None
        self.rate_limiter = RateLimiter(settings.market_rate_limit_seconds)

    def load(self, symbols: List[str], deadline: float | None = None):
        symbols = [sym for sym in symbols if sym]
        self.provenance = {sym: [] for sym in symbols}
        pending = set(symbols)
        start = "1900-01-01"
        end = "2100-01-01"

        def _record(sym: str, provider_name: str, cache_hit: bool, cache_age: float | None, success: bool):
            self.provenance.setdefault(sym, []).append(
                {
                    "provider": provider_name,
                    "endpoint": "prices",
                    "params": {"interval": "1d"},
                    "cache_hit": cache_hit,
                    "cache_age_hours": cache_age,
                    "success": success,
                }
            )

        def _cache_key(provider_name: str, symbol: str):
            if not self.cache:
                return None
            return self.cache.make_key(provider_name, "prices", symbol, start, end, {"interval": "1d"})

        def _cache_get_df(provider_name: str, symbol: str):
            if not self.cache:
                return None, None
            key = _cache_key(provider_name, symbol)
            payload, age = self.cache.get(key)
            if payload is None:
                return None, age
            return _records_to_df(payload), age

        def _cache_set_df(provider_name: str, symbol: str, df: pd.DataFrame):
            if not self.cache:
                return
            payload = _df_to_records(df)
            if payload is None:
                return
            key = _cache_key(provider_name, symbol)
            self.cache.set(key, payload)

        def _fetch_individual(provider_name: str, provider, symbols: list[str]):
            for sym in symbols:
                if sym not in pending:
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("time_budget_exceeded")
                self.rate_limiter.wait(deadline)

                def _call():
                    return provider.prices(sym, start=start, end=end)

                try:
                    df = retry_call(
                        _call,
                        attempts=settings.market_retry_attempts,
                        base_delay=settings.http_retry_backoff_seconds,
                        deadline=deadline,
                        retry_on_result=_retry_on_empty_df,
                    )
                except Exception:
                    df = None
                success = _df_is_usable(df)
                _record(sym, provider_name, cache_hit=False, cache_age=None, success=success)
                if success:
                    self.prices[sym] = df
                    pending.discard(sym)
                    _cache_set_df(provider_name, sym, df)

        providers = [
            ("yfinance", self.providers.get("yfinance")),
            ("yq", self.providers.get("yq")),
            ("stooq", self.providers.get("stooq")),
            ("openbb", self.providers.get("openbb")),
        ]
        deferred_by_provider: dict[str, list[str]] = {}

        # Phase 1: cache + batch attempts (no per-symbol retries)
        for name, provider in providers:
            if not provider or not pending:
                continue
            remaining = [sym for sym in symbols if sym in pending]
            if not remaining:
                continue

            # Cache-first
            if self.cache:
                for sym in list(remaining):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("time_budget_exceeded")
                    df, age = _cache_get_df(name, sym)
                    if df is None:
                        continue
                    success = _df_is_usable(df)
                    _record(sym, name, cache_hit=True, cache_age=age, success=success)
                    if success:
                        self.prices[sym] = df
                        pending.discard(sym)
                        remaining.remove(sym)

            if not remaining:
                continue

            if hasattr(provider, "prices_batch"):
                batch_size = max(1, settings.market_batch_size)
                for batch in _chunked(remaining, batch_size):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("time_budget_exceeded")
                    try:
                        self.rate_limiter.wait(deadline)

                        def _call():
                            return provider.prices_batch(batch, start=start, end=end)

                        results = retry_call(
                            _call,
                            attempts=1,
                            base_delay=settings.http_retry_backoff_seconds,
                            deadline=deadline,
                        )
                    except Exception:
                        results = None

                    if isinstance(results, dict):
                        for sym in batch:
                            if sym not in pending:
                                continue
                            df = results.get(sym)
                            success = _df_is_usable(df)
                            _record(sym, name, cache_hit=False, cache_age=None, success=success)
                            if success:
                                self.prices[sym] = df
                                pending.discard(sym)
                                _cache_set_df(name, sym, df)
                            else:
                                deferred_by_provider.setdefault(name, []).append(sym)
                    else:
                        deferred_by_provider.setdefault(name, []).extend(batch)
            else:
                deferred_by_provider.setdefault(name, []).extend(remaining)

        # Phase 2: individual attempts for remaining symbols
        for name, provider in providers:
            if not provider or not pending:
                continue
            deferred = deferred_by_provider.get(name) or []
            if not deferred:
                continue
            _fetch_individual(name, provider, deferred)

    def load_quotes(self, symbols: List[str], deadline: float | None = None):
        symbols = [sym for sym in symbols if sym]
        if not symbols:
            return
        if int(getattr(settings, "quote_ttl_minutes", 0) or 0) <= 0:
            return

        provider_name = None
        provider = None
        if self.providers.get("yq") and getattr(self.providers["yq"], "enabled", False):
            provider_name = "yq"
            provider = self.providers["yq"]
        elif self.providers.get("yfinance") and getattr(self.providers["yfinance"], "enabled", False):
            provider_name = "yfinance"
            provider = self.providers["yfinance"]
        if not provider or not hasattr(provider, "quotes_batch"):
            return

        def _to_utc_iso(ts, fallback: str) -> str:
            if ts is None:
                return fallback
            if isinstance(ts, (int, float)):
                return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
            if isinstance(ts, datetime):
                dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                return fallback

        quote_ttl_hours = max(float(settings.quote_ttl_minutes) / 60.0, 0.0)
        pending = set(symbols)

        def _cache_key(symbol: str):
            if not self.cache:
                return None
            return self.cache.make_key(provider_name, "quote", symbol, "", "", {"interval": "1h"})

        def _cache_get(symbol: str):
            if not self.cache or quote_ttl_hours <= 0:
                return None, None
            key = _cache_key(symbol)
            payload, age = self.cache.get(key)
            if not isinstance(payload, dict):
                return None, age
            if payload.get("price") is None:
                return None, age
            return payload, age

        def _cache_set(symbol: str, payload: dict):
            if not self.cache or quote_ttl_hours <= 0:
                return
            key = _cache_key(symbol)
            self.cache.set(key, payload, ttl_hours=quote_ttl_hours)

        if self.cache and quote_ttl_hours > 0:
            for sym in list(pending):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("time_budget_exceeded")
                payload, _age = _cache_get(sym)
                if payload:
                    self.quotes[sym] = payload
                    pending.discard(sym)

        if not pending:
            return

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("time_budget_exceeded")
        self.rate_limiter.wait(deadline)
        try:
            def _call():
                return provider.quotes_batch(sorted(pending))
            results = retry_call(
                _call,
                attempts=settings.market_retry_attempts,
                base_delay=settings.http_retry_backoff_seconds,
                deadline=deadline,
            )
        except Exception:
            results = {}

        fetched_at = now_utc_iso()
        if isinstance(results, dict):
            for sym in pending:
                payload = results.get(sym)
                if not isinstance(payload, dict):
                    continue
                price = payload.get("price")
                if price is None:
                    continue
                entry = {
                    "price": float(price),
                    "as_of_utc": _to_utc_iso(payload.get("timestamp"), fetched_at),
                    "fetched_at_utc": fetched_at,
                    "provider": provider_name,
                }
                self.quotes[sym] = entry
                _cache_set(sym, entry)
