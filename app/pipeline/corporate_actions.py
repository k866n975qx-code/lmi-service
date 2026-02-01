import sqlite3
import statistics
from datetime import date, datetime
import time
from typing import Iterable

import pandas as pd

from ..cache_layer import CacheLayer
from ..config import settings
from ..providers.nasdaq_calendar_adapter import NasdaqCalendarAdapter
from ..providers.openbb_adapter import OpenBBAdapter
from ..providers.yahooquery_adapter import YahooQueryAdapter
from ..providers.yfinance_adapter import YFinanceAdapter
from ..utils import now_utc_iso, RateLimiter, retry_call

DIV_DATE_CANDIDATES = [
    "ex_date",
    "exdate",
    "ex-dividend date",
    "ex_dividend_date",
    "exdividenddate",
    "date",
    "datetime",
    "timestamp",
]
PAY_DATE_CANDIDATES = ["pay_date", "paydate", "payment_date", "paymentdate"]
DIV_AMOUNT_CANDIDATES = ["dividend", "dividends", "amount", "cash_amount", "value"]
DIV_CURRENCY_CANDIDATES = ["currency", "currency_code", "iso_currency_code", "unofficial_currency_code"]
DIV_FREQUENCY_CANDIDATES = ["frequency", "freq", "dividend_frequency"]

SPLIT_DATE_CANDIDATES = ["ex_date", "exdate", "date", "datetime", "timestamp"]
SPLIT_RATIO_CANDIDATES = ["split", "splits", "ratio", "stock splits", "stock_splits", "value"]


def _pick_col(df: pd.DataFrame, candidates: list[str]):
    cols = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def _coerce_df(obj):
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        df = obj.to_frame("value")
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif hasattr(obj, "to_pandas"):
        df = obj.to_pandas()
    elif hasattr(obj, "to_df"):
        df = obj.to_df()
    else:
        return None
    if isinstance(df.index, pd.DatetimeIndex) and "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    return df


def _parse_ratio(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if ":" in text:
        left, right = text.split(":", 1)
        try:
            return float(left) / float(right)
        except ValueError:
            return None
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            return float(left) / float(right)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_date(val):
    if not val:
        return None
    text = str(val).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        return dt.date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _normalize_dividends(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        return []
    df = df.copy()
    date_col = _pick_col(df, DIV_DATE_CANDIDATES)
    pay_col = _pick_col(df, PAY_DATE_CANDIDATES)
    amt_col = _pick_col(df, DIV_AMOUNT_CANDIDATES)
    cur_col = _pick_col(df, DIV_CURRENCY_CANDIDATES)
    freq_col = _pick_col(df, DIV_FREQUENCY_CANDIDATES)

    if date_col:
        df["ex_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    if pay_col:
        df["pay_date"] = pd.to_datetime(df[pay_col], errors="coerce").dt.date
    if "ex_date" not in df.columns and "pay_date" in df.columns:
        df["ex_date"] = df["pay_date"]
    if amt_col:
        df["amount"] = pd.to_numeric(df[amt_col], errors="coerce")
    if cur_col:
        df["currency"] = df[cur_col]
    if freq_col:
        df["frequency"] = df[freq_col]

    out = []
    for row in df.itertuples(index=False):
        ex_date = getattr(row, "ex_date", None)
        amount = getattr(row, "amount", None)
        if ex_date is None or amount is None:
            continue
        record = {
            "symbol": symbol,
            "ex_date": str(ex_date),
            "pay_date": str(getattr(row, "pay_date", None)) if getattr(row, "pay_date", None) else None,
            "amount": float(amount),
            "currency": getattr(row, "currency", None),
            "frequency": getattr(row, "frequency", None),
        }
        out.append(record)
    return out


def _normalize_splits(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        return []
    df = df.copy()
    date_col = _pick_col(df, SPLIT_DATE_CANDIDATES)
    ratio_col = _pick_col(df, SPLIT_RATIO_CANDIDATES)

    if date_col:
        df["ex_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    if ratio_col:
        df["ratio"] = df[ratio_col].apply(_parse_ratio)

    out = []
    for row in df.itertuples(index=False):
        ex_date = getattr(row, "ex_date", None)
        ratio = getattr(row, "ratio", None)
        if ex_date is None or ratio is None:
            continue
        out.append({"symbol": symbol, "ex_date": str(ex_date), "ratio": float(ratio)})
    return out


def _provider_chain():
    providers = []
    if bool(settings.nasdaq_enable):
        providers.append(("nasdaq", NasdaqCalendarAdapter(enabled=True)))
    if bool(settings.yf_enable):
        providers.append(("yfinance", YFinanceAdapter(enabled=True)))
    if bool(settings.yq_enable):
        providers.append(("yahooquery", YahooQueryAdapter(enabled=True)))
    if bool(settings.providers_openbb):
        providers.append(("openbb", OpenBBAdapter(enabled=True)))
    return providers


def _fetch_events(provider, kind: str, symbol: str):
    if kind == "dividends" and hasattr(provider, "dividends"):
        return provider.dividends(symbol)
    if kind == "splits" and hasattr(provider, "splits"):
        return provider.splits(symbol)
    return None


def _fetch_with_cache(cache: CacheLayer | None, provider_name: str, kind: str, symbol: str, fetcher):
    if not cache:
        return fetcher(), False
    key = cache.make_key(provider_name, kind, symbol, None, None, {})
    payload, cache_hit, _cache_age = cache.fetch(key, fetcher)
    return payload, cache_hit


def _insert_dividends(
    conn: sqlite3.Connection,
    run_id: str,
    source: str,
    events: list[dict],
    table_name: str,
):
    if not events:
        return 0
    if table_name not in {"dividend_events_lm", "dividend_events_provider"}:
        raise ValueError("unexpected dividend table")
    cur = conn.cursor()
    fetched_at = now_utc_iso()
    count = 0
    for ev in events:
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {table_name} (
              symbol, ex_date, pay_date, amount, currency, frequency,
              source, run_id, fetched_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                ev["symbol"],
                ev["ex_date"],
                ev.get("pay_date"),
                ev["amount"],
                ev.get("currency"),
                ev.get("frequency"),
                source,
                run_id,
                fetched_at,
            ),
        )
        count += 1
    conn.commit()
    return count


def _insert_lm_dividends(conn: sqlite3.Connection, run_id: str, events: list[dict]):
    return _insert_dividends(conn, run_id, "lunchmoney", events, "dividend_events_lm")


def _insert_provider_dividends(conn: sqlite3.Connection, run_id: str, source: str, events: list[dict]):
    return _insert_dividends(conn, run_id, source, events, "dividend_events_provider")


def _insert_splits(conn: sqlite3.Connection, run_id: str, source: str, events: list[dict]):
    if not events:
        return 0
    cur = conn.cursor()
    fetched_at = now_utc_iso()
    count = 0
    for ev in events:
        cur.execute(
            """
            INSERT OR REPLACE INTO split_events (
              symbol, ex_date, ratio, source, run_id, fetched_at_utc
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                ev["symbol"],
                ev["ex_date"],
                ev["ratio"],
                source,
                run_id,
                fetched_at,
            ),
        )
        count += 1
    conn.commit()
    return count


def upsert_lm_dividend_events(conn: sqlite3.Connection, run_id: str):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT symbol, date, amount, currency
        FROM investment_transactions
        WHERE run_id=? AND transaction_type='dividend' AND symbol IS NOT NULL
        """,
        (run_id,),
    ).fetchall()
    events = []
    for symbol, date_str, amount, currency in rows:
        if not date_str or amount is None:
            continue
        events.append(
            {
                "symbol": symbol,
                "ex_date": date_str,
                "pay_date": date_str,
                "amount": abs(float(amount)),
                "currency": currency,
                "frequency": None,
            }
        )
    return _insert_lm_dividends(conn, run_id, events)


def load_provider_actions(conn: sqlite3.Connection, run_id: str, symbols: Iterable[str], deadline: float | None = None):
    providers = _provider_chain()
    cache = CacheLayer(
        settings.cache_dir,
        settings.cache_db_path,
        settings.cache_ttl_hours,
    ) if bool(settings.cache_enabled) else None
    rate_limiter = RateLimiter(settings.market_rate_limit_seconds)

    div_count = 0
    split_count = 0
    for sym in symbols:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("time_budget_exceeded")
        # Dividends
        for name, provider in providers:
            def _fetch():
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("time_budget_exceeded")
                rate_limiter.wait(deadline)
                def _call():
                    obj = _fetch_events(provider, "dividends", sym)
                    df = _coerce_df(obj)
                    return _normalize_dividends(sym, df)
                return retry_call(
                    _call,
                    attempts=settings.market_retry_attempts,
                    base_delay=settings.http_retry_backoff_seconds,
                    deadline=deadline,
                )
            events, _cache_hit = _fetch_with_cache(cache, name, "dividends", sym, _fetch)
            if events:
                div_count += _insert_provider_dividends(conn, run_id, name, events)
                break

        # Splits
        for name, provider in providers:
            def _fetch():
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("time_budget_exceeded")
                rate_limiter.wait(deadline)
                def _call():
                    obj = _fetch_events(provider, "splits", sym)
                    df = _coerce_df(obj)
                    return _normalize_splits(sym, df)
                return retry_call(
                    _call,
                    attempts=settings.market_retry_attempts,
                    base_delay=settings.http_retry_backoff_seconds,
                    deadline=deadline,
                )
            events, _cache_hit = _fetch_with_cache(cache, name, "splits", sym, _fetch)
            if events:
                split_count += _insert_splits(conn, run_id, name, events)
                break

    return {"dividends": div_count, "splits": split_count}


def symbols_for_actions(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT DISTINCT symbol FROM investment_transactions WHERE symbol IS NOT NULL"
    ).fetchall()
    return [row[0] for row in rows]


def estimate_dividend_schedule(
    conn: sqlite3.Connection,
    run_id: str,
    window_days: int = 14,
    ex_window_days: int = 35,
):
    cur = conn.cursor()
    lm_rows = cur.execute(
        """
        SELECT symbol, ex_date, amount
        FROM dividend_events_lm
        WHERE source='lunchmoney'
          AND (ex_date_est IS NULL OR pay_date_est IS NULL OR match_method IS NULL)
        """,
    ).fetchall()
    if not lm_rows:
        return 0

    prov_rows = cur.execute(
        """
        SELECT symbol, ex_date, pay_date, source
        FROM dividend_events_provider
        """
    ).fetchall()

    by_symbol: dict[str, list[dict]] = {}
    for symbol, ex_date, pay_date, source in prov_rows:
        sym = str(symbol).upper()
        ex_dt = _parse_date(ex_date)
        pay_dt = _parse_date(pay_date)
        if ex_dt is None and pay_dt is None:
            continue
        by_symbol.setdefault(sym, []).append(
            {"ex_date": ex_dt, "pay_date": pay_dt, "source": source}
        )

    updated = 0
    cur = conn.cursor()

    def _tighten_pay_window(symbol: str, default_window: int, min_samples: int = 3, floor: int = 1) -> int:
        rows = cur.execute(
            """
            SELECT match_days_delta
            FROM dividend_events_lm
            WHERE symbol=?
              AND match_source='nasdaq'
              AND match_method LIKE 'match_pay_date_within_%'
              AND match_days_delta IS NOT NULL
            """,
            (symbol,),
        ).fetchall()
        deltas = [abs(r[0]) for r in rows if r[0] is not None]
        if len(deltas) >= min_samples:
            median = statistics.median(deltas)
            return max(floor, min(default_window, int(round(median + 1))))
        rows = cur.execute(
            """
            SELECT match_days_delta
            FROM dividend_events_lm
            WHERE match_source='nasdaq'
              AND match_method LIKE 'match_pay_date_within_%'
              AND match_days_delta IS NOT NULL
            """
        ).fetchall()
        deltas = [abs(r[0]) for r in rows if r[0] is not None]
        if len(deltas) >= min_samples * 2:
            median = statistics.median(deltas)
            return max(floor, min(default_window, int(round(median + 1))))
        return default_window

    for symbol, cash_date_str, amount in lm_rows:
        sym = str(symbol).upper()
        cash_date = _parse_date(cash_date_str)
        if cash_date is None:
            continue
        candidates = by_symbol.get(sym, [])
        best = None
        pay_window = _tighten_pay_window(sym, window_days)
        pay_candidates = []
        ex_candidates = []
        for cand in candidates:
            pay_dt = cand["pay_date"]
            ex_dt = cand["ex_date"]
            if pay_dt is not None:
                delta = (pay_dt - cash_date).days
                if abs(delta) <= pay_window:
                    pay_candidates.append((abs(delta), delta, cand))
            if ex_dt is not None:
                delta = (ex_dt - cash_date).days
                if abs(delta) <= ex_window_days:
                    ex_candidates.append((abs(delta), delta, cand))

        if pay_candidates:
            _, delta, cand = sorted(pay_candidates, key=lambda item: (item[0], item[1]))[0]
            best = {
                "ex_date": cand["ex_date"],
                "pay_date": cand["pay_date"],
                "source": cand["source"],
                "delta": delta,
                "method": "match_pay_date_within_%dd" % pay_window,
            }
        elif ex_candidates:
            _, delta, cand = sorted(ex_candidates, key=lambda item: (item[0], item[1]))[0]
            best = {
                "ex_date": cand["ex_date"],
                "pay_date": cand["pay_date"],
                "source": cand["source"],
                "delta": delta,
                "method": "match_ex_date_within_%dd" % ex_window_days,
            }

        if best:
            method = best["method"]
            cur.execute(
                """
                UPDATE dividend_events_lm
                SET ex_date_est=?, pay_date_est=?, match_source=?, match_method=?, match_days_delta=?
                WHERE symbol=? AND ex_date=? AND amount=? AND source='lunchmoney'
                """,
                (
                    best["ex_date"].isoformat() if best["ex_date"] else None,
                    best["pay_date"].isoformat() if best["pay_date"] else cash_date.isoformat(),
                    best["source"],
                    method,
                    best["delta"],
                    sym,
                    cash_date_str,
                    amount,
                ),
            )
            updated += cur.rowcount

    conn.commit()
    return updated
